"""Agent that merges two OLAF ontologies into a new one: aligns them, then merges them.

Ported from OLAF's reference merging agent
(https://github.com/merlin-intelligence/olaf, ``demos/olaf_merging_agent/agent.py``),
adapted to:
  - stream progress through ``log()`` / ``set_progress()`` callbacks into
    :class:`cogmaps.ontology.store.OntologyJobStore` instead of stderr logging,
  - call Scaleway through ``litellm``'s native ``scaleway/`` provider the same
    way the other agents do (:mod:`cogmaps.ontology.reasoning_agent`),
  - hand the alignment (SSSOM TSV) and the final export to callbacks instead
    of writing to paths from a toml,
  - return a summary of the run for the page.

The OLAF session is the target ontology's (the server activates the ontology of
its config): it must have no content yet. The code drives the work; the LLM only
judges the candidate pairs:
  1. The reference ontology and the other one are imported into the target
     (``ontology_import``): the entities of the other one whose name the
     reference already uses are renamed, never merged silently.
  2. Matching — for each class, then property, then individual of the other
     ontology, the entities of the reference of the same kind that are
     semantically close (``min_score``) or have the same name are its candidates.
  3. The candidates are handed to the LLM, ``entities_per_task`` entities at a
     time, with their definitions, parents and source text. It answers for each
     pair: equivalent, narrower, broader or none, with a confidence.
  4. Merging — decisions with a confidence of at least ``min_confidence`` are
     applied: equivalent → ``concept_merge`` into the reference's entity;
     narrower / broader → rdfs:subClassOf (or rdfs:subPropertyOf) between them.
     Classes come first, so that properties merged afterwards have the same
     domain and range. A property correspondence is only applied when the
     domains and ranges allow it (a subproperty inherits its parent's).
  5. The alignment is handed over as an SSSOM TSV, between the URIs of the two
     source ontologies — applied or not — and the merged ontology is checked
     (``ontology_check``).
"""
from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass

from mcp import ClientSession

from cogmaps.core.llm_impacts import BACKGROUND_RETRY, acompletion_with_retry
from cogmaps.ontology.agent import select_seeds
from cogmaps.ontology.merging_prompts import SYSTEM_PROMPT, TASK_PROMPT
from cogmaps.ontology.olaf_config import has_ontology_content
from cogmaps.rag.llm_clients import strip_reasoning

SEARCH_TOP_K = 10           # semantic search results per entity (both ontologies' entities come back)
MIN_SCORE = 0.7             # min similarity for a reference entity to be a candidate
CANDIDATES_PER_ENTITY = 4   # candidates shown to the LLM per entity (same name first)
ENTITIES_PER_TASK = 8       # entities of the other ontology judged per LLM call
MIN_CONFIDENCE = 0.7        # below: written to the alignment, not applied
MAX_CHUNK_CHARS = 1200      # each source chunk's text is cut beyond this

RDF = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
RDFS = "http://www.w3.org/2000/01/rdf-schema#"
OWL = "http://www.w3.org/2002/07/owl#"
# Types that say what kind of term an entity is, not which class it belongs to.
_META_TYPES = (
    f"{OWL}Class", f"{OWL}NamedIndividual", f"{OWL}ObjectProperty", f"{OWL}DatatypeProperty",
    f"{OWL}Thing", f"{RDF}Property", f"{RDFS}Class",
)
# Matching order: properties after classes, whose merges re-point their domains and ranges.
KINDS = ("class", "property", "individual")
RELATIONS = {"equivalent", "narrower", "broader"}
# Alignment predicate, the reference entity as subject: "R narrowMatch O" = O is narrower than R.
SKOS_MATCH = {"equivalent": "skos:exactMatch", "narrower": "skos:narrowMatch", "broader": "skos:broadMatch"}


def _local(uri: str) -> str:
    return re.split(r"[#/]", uri.rstrip("#/"))[-1]


@dataclass
class Decision:
    other: dict
    reference: dict
    relation: str
    confidence: float
    reason: str
    applied: bool = False
    note: str = ""       # why it was not applied


# ─── Agent ────────────────────────────────────────────────────────────────────


class MergingAgent:
    def __init__(
        self,
        session: ClientSession,
        *,
        model: str,
        api_key: str,
        ontology_id: str,
        reference_id: str,
        other_id: str,
        log: Callable[[str], None],
        set_progress: Callable[[int, int], None],
        reference_collection: str | None = None,
        other_collection: str | None = None,
        alignment_cb: Callable[[str], None] | None = None,
        export_cb: Callable[[str], None] | None = None,
        min_confidence: float = MIN_CONFIDENCE,
        min_score: float = MIN_SCORE,
    ) -> None:
        self.session = session
        self.model = model
        self.api_key = api_key
        self.ontology_id = ontology_id
        self.graph = f"urn:olaf:{ontology_id}"
        self.reference_id = reference_id
        self.other_id = other_id
        self.collections = {"reference": reference_collection, "other": other_collection}
        self.log = log
        self.set_progress = set_progress
        self.alignment_cb = alignment_cb
        self.export_cb = export_cb
        self.min_confidence = min_confidence
        self.min_score = min_score

        self.reference: dict[str, dict] = {}   # uri in the merged ontology → entity, from import
        self.other: dict[str, dict] = {}
        self.same_name: dict[str, str] = {}    # other entity renamed on import → reference entity of that name
        self.decisions: list[Decision] = []

    # ── Tool calls ────────────────────────────────────────────────────────

    async def _call(self, tool_name: str, args: dict | None = None) -> str:
        """Tool call made by the code itself; an error stops the run."""
        result = await self.session.call_tool(tool_name, args or {})
        text = result.content[0].text if result.content else ""
        if text.startswith("Error"):
            raise RuntimeError(f"{tool_name} failed: {text}")
        return text

    async def _try_json(self, tool_name: str, args: dict):
        """Tool call made to gather context: an error only means less context."""
        try:
            return json.loads(await self._call(tool_name, args))
        except (RuntimeError, json.JSONDecodeError):
            return None

    # ── 1. Import ─────────────────────────────────────────────────────────

    async def prepare(self) -> None:
        existing = {o["id"] for o in json.loads(await self._call("ontology_list"))}
        if missing := [i for i in (self.reference_id, self.other_id) if i not in existing]:
            raise RuntimeError(f"Ontology not found: {', '.join(missing)}.")
        has_content = json.loads(await self._call("sparql_query", {"query": f"""
            PREFIX owl: <{OWL}>
            ASK {{ GRAPH <{self.graph}> {{
                ?e a ?t . VALUES ?t {{ owl:Class owl:NamedIndividual owl:ObjectProperty owl:DatatypeProperty }}
            }} }}"""}))
        if has_content.get("boolean"):
            raise RuntimeError(
                f"The target ontology '{self.ontology_id}' already has content: importing into it would "
                "duplicate it. Merge into a new ontology."
            )
        # The searches only need the two ontologies, now in the target.
        await select_seeds(self.session, [], self.log)

        for origin, source_id in (("reference", self.reference_id), ("other", self.other_id)):
            result = json.loads(await self._call("ontology_import", {"source_ontology_id": source_id}))
            if warning := result.get("warning"):
                raise RuntimeError(f"Import of '{source_id}': {warning} — semantic search is needed for matching.")
            entities = {e["uri"]: {**e, "origin": origin} for e in result["entities"]}
            self.log(f"Imported '{source_id}' ({origin}): {result['triples']} triples, {len(entities)} entities, "
                     f"{len(result['renamed'])} renamed (name already used).")
            if origin == "reference":
                self.reference = entities
            else:
                # Seed terms both reuse are the same entity already.
                self.other = {u: e for u, e in entities.items() if u not in self.reference}
                for old, new in result["renamed"].items():
                    clash = new.rsplit("#", 1)[0] + "#" + old.rsplit("#", 1)[1]
                    if clash in self.reference:
                        self.same_name[new] = clash

    # ── 2. Matching ───────────────────────────────────────────────────────

    async def candidates(self, kind: str) -> dict[str, list[dict]]:
        """Reference entities of the same kind each other entity may correspond to, best first:
        those of the same name, then the semantically closest above min_score."""
        tool = "property_semantic_search" if kind == "property" else "concept_semantic_search"
        found: dict[str, list[dict]] = {}
        for uri, e in self.other.items():
            if e["kind"] != kind:
                continue
            query = e["label"] + (f": {e['definition']}" if e.get("definition") else "")
            hits = json.loads(await self._call(tool, {"query": query, "top_k": SEARCH_TOP_K}))
            scores = {h["uri"]: h["score"] for h in hits}
            ranked = []
            if (same := self.same_name.get(uri)) and self.reference[same]["kind"] == kind:
                ranked.append({"uri": same, "score": scores.get(same), "same_name": True})
            for h in hits:
                ref = self.reference.get(h["uri"])
                if ref and ref["kind"] == kind and h["score"] >= self.min_score and h["uri"] != same:
                    ranked.append({"uri": h["uri"], "score": h["score"], "same_name": False})
            if ranked:
                found[uri] = ranked[:CANDIDATES_PER_ENTITY]
        return found

    # ── 3. Decisions ──────────────────────────────────────────────────────

    async def _parents(self, uris: list[str]) -> dict[str, list[str]]:
        """Labels of each entity's direct superclasses / superproperties / classes."""
        meta = ", ".join(f"<{t}>" for t in _META_TYPES)
        rows = (await self._try_json("sparql_query", {"limit": 1000, "query": f"""
            PREFIX rdf: <{RDF}> PREFIX rdfs: <{RDFS}>
            SELECT ?e ?p (SAMPLE(?l) AS ?label) WHERE {{ GRAPH <{self.graph}> {{
                VALUES ?e {{ {" ".join(f"<{u}>" for u in uris)} }}
                ?e rdfs:subClassOf|rdfs:subPropertyOf|rdf:type ?p .
                FILTER(isIRI(?p) && ?p NOT IN ({meta}))
                OPTIONAL {{ ?p rdfs:label ?l }}
            }} }} GROUP BY ?e ?p"""}) or {}).get("rows", [])
        out: dict[str, list[str]] = {}
        for r in rows:
            out.setdefault(r["e"], []).append(r["label"] or _local(r["p"]))
        return out

    async def _ends(self, uris: list[str]) -> dict[str, dict[str, dict[str, str]]]:
        """Current domains and ranges of properties, {uri: {"domain": {class: label}, "range": …}}:
        those of the merged ontology, where class merges have re-pointed them."""
        rows = (await self._try_json("sparql_query", {"limit": 1000, "query": f"""
            PREFIX rdfs: <{RDFS}>
            SELECT ?e ?end ?c (SAMPLE(?l) AS ?label) WHERE {{ GRAPH <{self.graph}> {{
                VALUES ?e {{ {" ".join(f"<{u}>" for u in uris)} }}
                ?e ?end ?c . FILTER(?end IN (rdfs:domain, rdfs:range) && isIRI(?c))
                OPTIONAL {{ ?c rdfs:label ?l }}
            }} }} GROUP BY ?e ?end ?c"""}) or {}).get("rows", [])
        out = {u: {"domain": {}, "range": {}} for u in uris}
        for r in rows:
            out[r["e"]]["domain" if r["end"] == f"{RDFS}domain" else "range"][r["c"]] = r["label"] or _local(r["c"])
        return out

    async def _implies(self, sub: dict[str, str], sup: dict[str, str]) -> bool:
        """Whether every class of `sup` is implied by one of `sub` (equal, or a superclass of it):
        a subject (object) valid for the property with ends `sub` is then valid for `sup`."""
        for b in sup:
            if b == f"{OWL}Thing" or b in sub:
                continue
            found = False
            for a in sub:
                result = await self._try_json("sparql_query", {"query": f"""
                    PREFIX rdfs: <{RDFS}>
                    ASK {{ GRAPH <{self.graph}> {{ <{a}> rdfs:subClassOf+ <{b}> }} }}"""})
                if (result or {}).get("boolean"):
                    found = True
                    break
            if not found:
                return False
        return True

    async def _incompatible(self, d: Decision) -> str | None:
        """Why a property correspondence cannot hold given the domains and ranges, or None.
        A subproperty inherits its parent's domain and range, and two merged properties keep
        both, which OWL reads as their intersection: they must match."""
        o, r = d.other["uri"], d.reference["uri"]
        ends = await self._ends([o, r])
        for end in ("domain", "range"):
            sub, sup = ends[o][end], ends[r][end]
            if (d.relation in ("equivalent", "narrower") and not await self._implies(sub, sup)) or \
                    (d.relation in ("equivalent", "broader") and not await self._implies(sup, sub)):
                return (f"{end} {', '.join(sub.values()) or '(none)'} of '{d.other['label']}' vs "
                        f"{', '.join(sup.values()) or '(none)'} of '{d.reference['label']}'")
        return None

    @staticmethod
    def _describe(e: dict, parents: list[str], ends: dict[str, dict[str, str]] | None = None) -> str:
        text = f'"{e["label"]}" `{e["uri"]}`'
        if definition := e.get("definition"):
            text += " — " + (definition[:400] + "…" if len(definition) > 400 else definition)
        details = []
        if parents:
            details.append(f"parents: {', '.join(parents)}")
        if e["kind"] == "property":
            details.append(f"{e.get('property_type') or ''} property".strip())
            if ends is not None:
                details.append(" → ".join(", ".join(ends[end].values()) or "(any)" for end in ("domain", "range")))
        if e.get("source_chunk_id"):
            details.append(f"source chunk: {e['source_chunk_id']}")
        return text + (f" [{'; '.join(details)}]" if details else "")

    async def _sources(self, entities: list[dict]) -> str:
        """Text of the entities' source chunks, read from their ontology's chunk collection."""
        texts: dict[str, str] = {}
        for origin, collection in self.collections.items():
            ids = list(dict.fromkeys(e["source_chunk_id"] for e in entities
                                     if e["origin"] == origin and e.get("source_chunk_id")))
            if not ids or not collection:
                continue
            if await self._try_json("chunk_collection_switch", {"collection": collection}) is None:
                continue
            for c in await self._try_json("chunk_read_batch", {"chunk_ids": ids}) or []:
                if text := re.sub(r"[ \t]+", " ", c.get("text") or ""):
                    texts[c["id"]] = text[:MAX_CHUNK_CHARS] + (" …" if len(text) > MAX_CHUNK_CHARS else "")
        return "\n\n".join(f"### Chunk {cid}\n{text}" for cid, text in texts.items()) or "(no source text)"

    async def _ask(self, task: str, label: str) -> list[dict]:
        """The LLM's decisions for one task; asked once more if its answer is not valid JSON."""
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": task}]
        for attempt in (1, 2):
            response = await acompletion_with_retry(
                retry=BACKGROUND_RETRY,
                log=self.log,
                model=f"scaleway/{self.model}",
                api_key=self.api_key,
                messages=messages,
                max_tokens=4096,
                temperature=0,
                timeout=120,
            )
            text = strip_reasoning(response.choices[0].message.content or "").strip()
            match = re.search(r"\{.*\}", text, re.DOTALL)
            try:
                return json.loads(match.group(0) if match else text)["decisions"]
            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                self.log(f"[{label}] invalid answer ({exc}), attempt {attempt}: {text[:300]}")
                messages += [{"role": "assistant", "content": text},
                             {"role": "user", "content": 'Answer with the JSON object only: {"decisions": [...]}'}]
        return []

    async def decide(self, kind: str, found: dict[str, list[dict]]) -> list[Decision]:
        items = list(found.items())
        tasks = [items[i : i + ENTITIES_PER_TASK] for i in range(0, len(items), ENTITIES_PER_TASK)]
        decisions: list[Decision] = []
        for n, group in enumerate(tasks, 1):
            label = f"{kind} · task {n}/{len(tasks)}"
            uris = list(dict.fromkeys(u for o, cands in group for u in [o, *(c["uri"] for c in cands)]))
            parents = await self._parents(uris)
            ends = await self._ends(uris) if kind == "property" else {}
            pairs: dict[str, tuple[dict, dict]] = {}  # candidate id (R1.2) → (other entity, reference entity)
            blocks = []
            for i, (o, cands) in enumerate(group, 1):
                other = self.other[o]
                lines = [f"### O{i}. {self._describe(other, parents.get(o, []), ends.get(o))}", "Candidates:"]
                for j, c in enumerate(cands, 1):
                    ref = self.reference[c["uri"]]
                    pairs[f"R{i}.{j}"] = (other, ref)
                    how = ", ".join((["same name"] if c["same_name"] else [])
                                    + ([f"score {c['score']:.2f}"] if c["score"] is not None else []))
                    lines.append(f"- R{i}.{j} ({how}) {self._describe(ref, parents.get(c['uri'], []), ends.get(c['uri']))}")
                blocks.append("\n".join(lines))
            involved = [self.other[o] for o, _ in group] + [self.reference[c["uri"]] for _, cands in group for c in cands]
            answer = await self._ask(
                TASK_PROMPT.format(count=len(group), kind=kind, entities="\n\n".join(blocks),
                                   sources=await self._sources(involved)),
                label,
            )

            equivalent: dict[str, Decision] = {}
            for d in answer:
                relation = str(d.get("relation", "none")).lower()
                if relation not in RELATIONS:
                    continue
                # The candidate id says which pair it is (R1.2 is a candidate of O1).
                pair = pairs.get(str(d.get("reference")))
                if not pair or (kind == "individual" and relation != "equivalent"):
                    self.log(f"[{label}] decision ignored (unknown candidate, or relation not possible): {d}")
                    continue
                try:
                    confidence = float(d.get("confidence", 0))
                except (TypeError, ValueError):
                    confidence = 0.0
                decision = Decision(pair[0], pair[1], relation, confidence, str(d.get("reason", "")))
                if relation == "equivalent":  # at most one per entity: the most certain
                    best = equivalent.get(pair[0]["uri"])
                    if best and best.confidence >= confidence:
                        continue
                    equivalent[pair[0]["uri"]] = decision
                else:
                    decisions.append(decision)
            decisions += equivalent.values()
            self.log(f"[{label}] {sum(1 for d in decisions if d.other['uri'] in {o for o, _ in group})} "
                     f"correspondence(s) found for {len(group)} entities.")
        return decisions

    # ── 4. Merge ──────────────────────────────────────────────────────────

    async def apply(self, d: Decision) -> None:
        o, r = d.other["uri"], d.reference["uri"]
        sub_property = f"{RDFS}subPropertyOf" if d.other["kind"] == "property" else f"{RDFS}subClassOf"
        if d.relation == "equivalent":
            await self._call("concept_merge", {"keep_uri": r, "merge_uri": o})
        elif d.relation == "narrower":
            await self._call("relation_add", {"subject_uri": o, "property_uri": sub_property, "object_value": r})
        else:
            await self._call("relation_add", {"subject_uri": r, "property_uri": sub_property, "object_value": o})
        d.applied = True
        self.log(f"Applied: {d.other['label']} {d.relation} {d.reference['label']} ({d.confidence:.2f}) — {d.reason}")

    async def merge(self, decisions: list[Decision]) -> None:
        # Equivalences first: a subsumption between an entity and one it is merged with is dropped.
        merged = {}
        for d in sorted(decisions, key=lambda d: d.relation != "equivalent"):
            if d.confidence < self.min_confidence:
                d.note = f"confidence below {self.min_confidence:.2f}"
                self.log(f"Not applied (confidence {d.confidence:.2f} < {self.min_confidence:.2f}): "
                         f"{d.other['label']} {d.relation} {d.reference['label']} — {d.reason}")
                continue
            if d.relation != "equivalent" and merged.get(d.other["uri"]):
                d.note = "merged with another entity"
                continue
            if d.other["kind"] == "property" and (problem := await self._incompatible(d)):
                d.note = f"incompatible {problem}"
                self.log(f"Not applied ({d.note}): {d.other['label']} {d.relation} {d.reference['label']} — {d.reason}")
                continue
            try:
                await self.apply(d)
            except RuntimeError as exc:
                d.note = "failed to apply"
                self.log(f"Not applied: {d.other['label']} {d.relation} {d.reference['label']} — {exc}")
                continue
            if d.relation == "equivalent":
                merged[d.other["uri"]] = d.reference["uri"]

    # ── 5. Output ─────────────────────────────────────────────────────────

    def alignment_tsv(self) -> str:
        """SSSOM TSV, between the URIs the entities have in their source ontologies."""
        header = [
            "# curie_map:",
            "#   skos: http://www.w3.org/2004/02/skos/core#",
            "#   semapv: https://w3id.org/semapv/vocab/",
            "# mapping_tool: cogmaps merging agent (olaf_merging_agent)",
            f"# subject_source: urn:olaf:{self.reference_id}",
            f"# object_source: urn:olaf:{self.other_id}",
            "subject_id\tsubject_label\tpredicate_id\tobject_id\tobject_label\tmapping_justification\t"
            "confidence\tapplied\tcomment",
        ]

        def clean(s: str) -> str:
            return " ".join(str(s).split())

        rows = [
            "\t".join([
                d.reference.get("source_uri", d.reference["uri"]), clean(d.reference["label"]), SKOS_MATCH[d.relation],
                d.other.get("source_uri", d.other["uri"]), clean(d.other["label"]), "semapv:CompositeMatching",
                f"{d.confidence:.2f}", str(d.applied).lower(), clean(d.reason + (f" [not applied: {d.note}]" if d.note else "")),
            ])
            for d in self.decisions
        ]
        return "\n".join(header + rows) + "\n"

    async def final_check(self) -> dict:
        report = json.loads(await self._call("ontology_check"))
        r, it = report.get("reasoner", {}), report.get("integrity", {})
        if error := r.get("error"):
            self.log(f"Reasoner unavailable, only the SPARQL checks ran: {error}")
        counts = {
            "consistent": r.get("consistent"),
            "unsatisfiable": len(r.get("unsatisfiable_classes") or []),
            "cycles": len(it.get("subclass_cycles", [])),
            "domain": len(it.get("domain_violations", [])),
            "range": len(it.get("range_violations", [])),
            "issues": report.get("issues", 0),
        }
        self.log(
            f"Check of the merged ontology — consistent: {counts['consistent']}, unsatisfiable: "
            f"{counts['unsatisfiable']}, classes in cycles: {counts['cycles']}, domain: {counts['domain']}, "
            f"range: {counts['range']}, issues: {counts['issues']}"
            + (" — run the reasoning on it to fix them" if counts["issues"] else "")
        )
        return counts

    # ── Run ───────────────────────────────────────────────────────────────

    async def run(self) -> dict:
        """Import → match and merge classes, properties, individuals → check. Returns what the
        run did: ``imported`` (entities per source), ``equivalent`` / ``subsumptions`` (applied),
        ``not_applied``, ``correspondences`` (one row per decision) and ``check``."""
        steps = ["import", *KINDS, "check"]

        def step(name: str) -> None:
            self.set_progress(steps.index(name) + 1, len(steps))
            self.log(f"=== Step {steps.index(name) + 1}/{len(steps)}: {name} ===")

        step("import")
        await self.prepare()
        for kind in KINDS:
            step(kind)
            found = await self.candidates(kind)
            alone = sum(1 for e in self.other.values() if e["kind"] == kind) - len(found)
            self.log(f"Matching {kind}: {len(found)} entities with candidates, {alone} without (kept as they are).")
            if not found:
                continue
            decisions = await self.decide(kind, found)
            await self.merge(decisions)
            self.decisions += decisions

        applied = [d for d in self.decisions if d.applied]
        summary = {
            "imported": {"reference": len(self.reference), "other": len(self.other)},
            "equivalent": sum(d.relation == "equivalent" for d in applied),
            "subsumptions": sum(d.relation != "equivalent" for d in applied),
            "not_applied": len(self.decisions) - len(applied),
            "correspondences": [
                {
                    "kind": d.other["kind"], "other": d.other["label"], "relation": d.relation,
                    "reference": d.reference["label"], "confidence": round(d.confidence, 2),
                    "applied": d.applied, "reason": d.reason + (f" [not applied: {d.note}]" if d.note else ""),
                }
                for d in self.decisions
            ],
        }
        self.log(f"Merge done: {summary['equivalent']} equivalences merged, {summary['subsumptions']} "
                 f"subsumptions added, {summary['not_applied']} correspondences not applied.")
        if self.alignment_cb:
            self.alignment_cb(self.alignment_tsv())
            self.log(f"Alignment ({len(self.decisions)} correspondences) saved.")

        step("check")
        try:
            summary["check"] = await self.final_check()
        except (RuntimeError, json.JSONDecodeError) as exc:  # the merge itself is done
            self.log(f"Check of the merged ontology failed: {exc}")
            summary["check"] = None
        if self.export_cb:
            content = await self._call("ontology_export", {"include_seeds": False})
            if has_ontology_content(content):
                self.export_cb(content)
        return summary


async def run_merge(session: ClientSession, **kwargs) -> dict:
    """Run :class:`MergingAgent` against an already-initialized OLAF session on the target
    ontology. ``kwargs`` are :class:`MergingAgent`'s keyword arguments. Returns what the run
    did (see :meth:`MergingAgent.run`)."""
    return await MergingAgent(session, **kwargs).run()
