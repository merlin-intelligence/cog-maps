"""Agent that curates an existing OLAF ontology: dedup, repair, enrich, infer.

Ported from OLAF's reference reasoning agent
(https://github.com/merlin-intelligence/olaf, ``demos/olaf_reasoning_agent/agent.py``),
adapted to:
  - stream progress through ``log()`` / ``set_progress()`` callbacks into
    :class:`cogmaps.ontology.store.OntologyJobStore` instead of stderr logging,
  - call Scaleway through ``litellm``'s native ``scaleway/`` provider the same
    way the building agent does (:mod:`cogmaps.ontology.agent`),
  - hand the pre-change Turtle backup and the final export to callbacks
    instead of writing to paths from a toml,
  - return a summary of the run (problem counts of every check, merges,
    disjointness, inferences) for the page.

The code drives the work; the LLM only decides and applies the changes:
  0. Deduplication — classes whose concept embeddings are closer than
     ``dedup_threshold`` are reviewed by pairs: merge or keep.
  1. ``ontology_check`` (Pellet: inconsistency, unsatisfiable classes; SPARQL:
     subclass cycles, domain/range violations, untyped individuals, property
     misuse, unknown vocabulary terms) and ``ontology_orphans`` run.
  2. The problems are split into groups of ``PROBLEMS_PER_TASK``. For each
     group the code gathers the context itself — the axioms involved, the
     entities with their definitions and ancestors, the classes a property
     could be widened to, the source chunk texts — and hands it to a fresh LLM
     conversation, which fixes them with a few write tools.
  3. The check runs again, for up to ``max_rounds`` rounds, until nothing is
     left or nothing changes.
  4. Enrichment (once the ontology is consistent) — groups of sibling classes
     with pairs not declared disjoint are handed to the LLM, which declares
     those that exclude each other. The check runs again: a disjointness that
     contradicts an axiom shows up, flagged as the likely culprit.
  5. Inference (once the ontology is consistent) — the reasoner's inferences
     not materialized yet are reviewed by the LLM, each with why it holds (an
     absurd one reveals a wrong axiom, which it fixes), the check runs again,
     then the inferences are written into the ontology, marked as inferred.
"""
from __future__ import annotations

import asyncio
import json
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import litellm
from mcp import ClientSession

from cogmaps.core.llm_impacts import tracked_completion
from cogmaps.ontology.agent import mcp_tools_to_litellm
from cogmaps.ontology.olaf_config import has_ontology_content
from cogmaps.ontology.reasoning_prompts import (
    DEDUP_PROMPT,
    ENRICH_PROMPT,
    FIX_PROMPT,
    REVIEW_PROMPT,
    SYSTEM_PROMPT,
)
from cogmaps.rag.llm_clients import strip_reasoning

# Tools the LLM may use: the writes, plus targeted reads for what the task context does not
# cover (e.g. finding where to attach an orphan). Everything else is hidden and refused.
WRITE_TOOLS = frozenset({
    "relation_delete", "relation_add", "property_update", "concept_update", "concept_merge",
    "restriction_delete", "entity_delete", "disjoint_add",
})
FIX_TOOLS = WRITE_TOOLS | {"concept_get", "property_get", "relation_search", "concept_search"}

MAX_ROUNDS = 3
PROBLEMS_PER_TASK = 5
DEDUP_THRESHOLD = 0.9          # min semantic similarity for a pair of classes to be reviewed
DEDUP_PAIRS_PER_TASK = 20
SIBLING_GROUPS_PER_TASK = 4
MAX_SIBLINGS_PER_GROUP = 15    # larger sibling sets (e.g. the root classes) are split
INFERENCES_PER_TASK = 10
MAX_EXPLAINED_INFERENCES = 20  # explained by the reasoner (one JVM run each); taxonomic ones are free
MAX_ITERATIONS = 12            # LLM ↔ tool rounds per task
MAX_SOURCE_CHUNKS = 8          # source chunks per task (those of the axioms at stake first)
MAX_CHUNK_CHARS = 2500         # each chunk's text is cut beyond this
MAX_ENTITIES = 25              # entities described per task
MAX_TOOL_RESULT_CHARS = 20_000
KEEP_RECENT_TURNS = 6          # the last N LLM turns keep their full tool results
PRUNED_RESULT_CHARS = 500      # older tool results are cut to this preview
RATE_LIMIT_RETRIES = 6
TRANSIENT_RETRIES = 2

RDF_TYPE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
RDFS = "http://www.w3.org/2000/01/rdf-schema#"
OWL = "http://www.w3.org/2002/07/owl#"
# Types that say what kind of term an entity is, not which class it belongs to.
_META_TYPES = (
    "http://www.w3.org/2002/07/owl#Class", "http://www.w3.org/2002/07/owl#NamedIndividual",
    "http://www.w3.org/2002/07/owl#ObjectProperty", "http://www.w3.org/2002/07/owl#DatatypeProperty",
    "http://www.w3.org/2002/07/owl#Thing",
)

# Errors worth retrying as-is: the provider was slow or briefly unavailable.
_TRANSIENT_ERRORS = (
    litellm.Timeout,
    litellm.APIConnectionError,
    litellm.InternalServerError,
    litellm.ServiceUnavailableError,
)


def _completion(log: Callable[[str], None], **kwargs):
    """``tracked_completion``, retried on rate limits (HTTP 429) with a growing wait — quotas
    are usually per minute, so the wait goes up to a full minute — and on transient errors
    (timeout, connection, 5xx), retried a few times after a short pause."""
    rate_limited = transient = 0
    while True:
        try:
            return tracked_completion(**kwargs)
        except litellm.RateLimitError:
            if rate_limited == RATE_LIMIT_RETRIES:
                raise
            wait = min(15 * 2 ** rate_limited, 60)
            rate_limited += 1
            log(f"Rate limited by the LLM provider — retrying in {wait}s ({rate_limited}/{RATE_LIMIT_RETRIES}).")
            time.sleep(wait)
        except _TRANSIENT_ERRORS as exc:
            if transient == TRANSIENT_RETRIES:
                raise
            transient += 1
            log(f"LLM call failed ({type(exc).__name__}) — retrying in 10s ({transient}/{TRANSIENT_RETRIES}).")
            time.sleep(10)


def prune_history(messages: list[dict], keep_recent_turns: int, pruned_chars: int) -> list[dict]:
    """Copy of ``messages`` to send to the LLM: tool results older than the last
    ``keep_recent_turns`` turns (assistant message + its tool results) are cut to a
    ``pruned_chars`` preview. The task message, which holds the context, is never pruned."""
    starts = [i for i, m in enumerate(messages) if m["role"] == "assistant" and m.get("tool_calls")]
    keep = max(1, keep_recent_turns)
    cutoff = starts[-keep] if len(starts) > keep else 0
    pruned = []
    for i, m in enumerate(messages):
        content = m.get("content") or ""
        if i < cutoff and m["role"] == "tool" and len(content) > pruned_chars:
            m = {**m, "content": (
                content[:pruned_chars]
                + f"\n… [pruned from context: {len(content)} chars — call the tool again if you need it]"
            )}
        pruned.append(m)
    return pruned


def _local(uri: str) -> str:
    return re.split(r"[#/]", uri.rstrip("#/"))[-1]


def _summarise(args: dict) -> str:
    """One-line representation of tool arguments for logging. URIs are kept whole, so that
    every change can be read (and undone) from the log; long texts are shortened."""
    parts = []
    for k, v in args.items():
        s = " ".join(str(v).split())
        parts.append(f"{k}={s[:120]!r}…" if len(s) > 120 and not s.startswith("http") else f"{k}={s!r}")
    return ", ".join(parts)


def report_counts(report: dict, orphans: list[dict]) -> dict:
    """Problem counts of one check, as shown in the log and on the page."""
    r, it = report.get("reasoner", {}), report.get("integrity", {})
    return {
        "consistent": r.get("consistent"),
        "reasoner_error": r.get("error"),
        "unsatisfiable": len(r.get("unsatisfiable_classes") or []),
        "cycles": len(it.get("subclass_cycles", [])),
        "domain": len(it.get("domain_violations", [])),
        "range": len(it.get("range_violations", [])),
        "untyped": len(it.get("untyped_individuals", [])),
        "kind": len(it.get("property_kind_mismatches", [])),
        "unknown_terms": len(it.get("unknown_terms", [])),
        "orphans": len(orphans),
    }


def _format_counts(label: str, c: dict) -> str:
    consistent = "?" if c["consistent"] is None else c["consistent"]
    return (
        f"{label} — consistent: {consistent}, unsatisfiable: {c['unsatisfiable']}, "
        f"classes in cycles: {c['cycles']}, domain: {c['domain']}, range: {c['range']}, "
        f"untyped: {c['untyped']}, kind: {c['kind']}, unknown terms: {c['unknown_terms']}, "
        f"orphans: {c['orphans']}"
    )


# ─── Problems ─────────────────────────────────────────────────────────────────


@dataclass
class Problem:
    kind: str
    text: str                                                       # markdown description for the LLM
    entities: list[str] = field(default_factory=list)               # URIs to describe
    triples: list[tuple[str, str, str]] = field(default_factory=list)  # axioms whose sources to show
    violation: dict | None = None                                   # domain/range: the check's row

    @property
    def key(self) -> tuple:
        """Identity across rounds, to tell whether a round changed anything."""
        return (self.kind, self.text)


# ─── Agent ────────────────────────────────────────────────────────────────────


class ReasoningAgent:
    def __init__(
        self,
        session: ClientSession,
        mcp_tools: list,
        *,
        model: str,
        api_key: str,
        ontology_id: str,
        log: Callable[[str], None],
        set_progress: Callable[[int, int], None],
        backup_cb: Callable[[str], None] | None = None,
        export_cb: Callable[[str], None] | None = None,
        max_rounds: int = MAX_ROUNDS,
        fix_orphans: bool = True,
        include_seeds: bool = False,
        dedup: bool = True,
        enrich_disjointness: bool = True,
        infer: bool = True,
        review_inferences: bool = True,
    ) -> None:
        self.session = session
        self.model = model
        self.api_key = api_key
        self.ontology_id = ontology_id
        self.graph = f"urn:olaf:{ontology_id}"
        self.log = log
        self.set_progress = set_progress
        self.backup_cb = backup_cb
        self.export_cb = export_cb
        self.max_rounds = max_rounds
        self.fix_orphans = fix_orphans
        self.include_seeds = include_seeds
        self.dedup = dedup
        self.enrich_disjointness = enrich_disjointness
        self.infer = infer
        self.review_inferences = review_inferences

        # With an older OLAF, the tools it lacks are simply not offered.
        self.tools = mcp_tools_to_litellm([t for t in mcp_tools if t.name in FIX_TOOLS])
        self.system_prompt = SYSTEM_PROMPT.format(graph=self.graph)
        self._entity_cache: dict[str, dict | None] = {}
        self._backed_up = False
        # Problems the last check left unfixed: not handed over again unless something changed.
        self._unresolved: set | None = None
        # Disjointness axioms the enrichment declared in this run, as stored (a disjointWith b):
        # flagged as the likely culprit when they appear in an explanation.
        self._added_disjoint: set[tuple[str, str]] = set()

        # What the run did, for the page (see run()).
        self.checks: list[dict] = []
        self.changes = 0  # successful write tool calls by the LLM
        self.dedup_pairs = 0
        self.inferences: dict | None = None

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

    async def _llm_tool_call(self, tool_name: str, raw_args: str) -> str:
        """Tool call requested by the LLM; errors are returned to it so it can adapt."""
        try:
            tool_args = json.loads(raw_args) if raw_args else {}
        except json.JSONDecodeError:
            return f"Error: invalid JSON arguments: {raw_args}"
        if tool_name not in FIX_TOOLS:
            self.log(f"Refused tool: {tool_name}")
            return f"Error: {tool_name} is not available for this task."

        self.log(f"Tool call: {tool_name}({_summarise(tool_args)})")
        try:
            result = await self.session.call_tool(tool_name, tool_args)
            content = result.content[0].text if result.content else ""
        except Exception as exc:  # noqa: BLE001
            self.log(f"  tool {tool_name} failed: {exc}")
            return f"Error: {exc}"

        if tool_name in WRITE_TOOLS and not content.startswith("Error"):
            self.changes += 1
        self.log(f"  -> {content[:200]}")
        if len(content) > MAX_TOOL_RESULT_CHARS:
            content = content[:MAX_TOOL_RESULT_CHARS] + f"\n… [truncated: {len(content)} chars in total — narrow your query]"
        return content

    async def converse(self, task: str, label: str) -> str:
        """One task = one fresh conversation: the LLM calls tools until it replies with a
        plain-text summary. Returns that summary."""
        messages: list[dict] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": task},
        ]
        for iteration in range(1, MAX_ITERATIONS + 1):
            self.log(f"[{label}] iteration {iteration}")
            response = await asyncio.to_thread(
                _completion,
                self.log,
                model=f"scaleway/{self.model}",
                api_key=self.api_key,
                messages=prune_history(messages, KEEP_RECENT_TURNS, PRUNED_RESULT_CHARS),
                tools=self.tools,
                tool_choice="auto",
                # Same litellm workaround as cogmaps.ontology.agent.run_build: the
                # "scaleway" provider doesn't whitelist tools/tool_choice.
                allowed_openai_params=["tools", "tool_choice"],
                max_tokens=4096,
                temperature=0,
                timeout=120,
            )
            msg = response.choices[0].message
            text = strip_reasoning(msg.content or "")

            if not msg.tool_calls:
                self.log(f"[{label}] done: {text[:1000]}")
                return text
            if text:
                self.log(f"[{label}] {text[:500]}")

            messages.append({
                "role": "assistant",
                "content": msg.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    }
                    for tc in msg.tool_calls
                ],
            })
            for tc in msg.tool_calls:
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "name": tc.function.name,
                    "content": await self._llm_tool_call(tc.function.name, tc.function.arguments),
                })

        self.log(f"[{label}] reached max iterations ({MAX_ITERATIONS}) — task stopped.")
        return ""

    # ── Problems from the check report ────────────────────────────────────

    async def _cycle_edges(self, classes: list[str]) -> list[tuple[str, str, str]]:
        members = ", ".join(f"<{c}>" for c in classes)
        rows = await self._try_json("sparql_query", {"query": f"""
            PREFIX rdfs: <{RDFS}>
            SELECT ?a ?b WHERE {{ GRAPH <{self.graph}> {{
                ?a rdfs:subClassOf ?b . FILTER(?a IN ({members}) && ?b IN ({members}))
            }} }}"""})
        return [(r["a"], RDFS + "subClassOf", r["b"]) for r in (rows or {}).get("rows", [])]

    async def problems(self, report: dict, orphans: list[dict]) -> list[Problem]:
        out: list[Problem] = []
        reasoner = report.get("reasoner", {})

        def entity_uris(entities: dict) -> list[str]:
            uris = []
            for v in entities.values():
                uris.extend(v if isinstance(v, list) else [v])
            return uris

        added = {frozenset((_local(a), _local(b))): (a, b) for a, b in self._added_disjoint}

        def explanation(expl: list[list[str]]) -> str:
            lines = []
            for axiom in (a for axioms in expl[:1] for a in axioms):
                lines.append(f"  - `{axiom}`")
                m = re.fullmatch(r"(\S+) disjointWith (\S+)", axiom.strip())
                if m and (pair := added.get(frozenset(m.groups()))):
                    lines.append(
                        f"    ⚠ declared by the enrichment of this run — the likely wrong axiom: undo with "
                        f"relation_delete(subject_uri=`{pair[0]}`, property_uri=`{OWL}disjointWith`, "
                        f"object_value=`{pair[1]}`) unless the text says they exclude each other"
                    )
            return "\n".join(lines)

        if inc := reasoner.get("inconsistency"):
            out.append(Problem(
                "inconsistency",
                f"**Inconsistency** — {inc.get('reason', '')}\nExplanation (minimal axioms causing it):\n"
                + explanation(inc.get("explanations", [])),
                entities=entity_uris(reasoner.get("entities", {})),
            ))
        for u in reasoner.get("unsatisfiable_classes", []):
            names = {w for axioms in u["explanations"][:1] for a in axioms for w in re.findall(r"[A-Za-z_][\w.-]*", a)}
            out.append(Problem(
                "unsatisfiable",
                f"**Unsatisfiable class** `{u['uri']}` — it can have no instance.\nExplanation:\n"
                + explanation(u["explanations"]),
                entities=[uri for n, v in reasoner.get("entities", {}).items() if n in names
                          for uri in (v if isinstance(v, list) else [v])],
            ))

        integrity = report.get("integrity", {})
        if cycle := integrity.get("subclass_cycles"):
            edges = await self._cycle_edges(cycle)
            out.append(Problem(
                "cycle",
                "**Subclass cycle** between: " + ", ".join(f"`{c}`" for c in cycle)
                + "\nSubclass axioms among them:\n" + "\n".join(f"  - `{a}` rdfs:subClassOf `{b}`" for a, _, b in edges),
                entities=list(cycle),
                triples=edges,
            ))
        for kind, key in (("domain", "domain_violations"), ("range", "range_violations")):
            for v in integrity.get(key, []):
                side = "subject" if kind == "domain" else "object"
                out.append(Problem(
                    kind,
                    f"**{kind.capitalize()} violation** — `{v['subject']}` `{v['property']}` `{v['object']}`: "
                    f"the {side} is not a `{v['expected']}` ({kind} of the property).",
                    entities=[v["subject"], v["property"], v["object"], v["expected"]],
                    triples=[(v["subject"], v["property"], v["object"])],
                    violation=v,
                ))
        for v in integrity.get("property_kind_mismatches", []):
            out.append(Problem(
                "kind",
                f"**Property kind mismatch** — `{v['subject']}` `{v['property']}` `{v['object']}`: {v['problem']}.",
                entities=[v["subject"], v["property"]],
                triples=[(v["subject"], v["property"], v["object"])],
            ))
        for v in integrity.get("unknown_terms", []):
            hint = f" — the closest real term is `{v['suggestion']}`" if v.get("suggestion") else ""
            out.append(Problem(
                "unknown_term",
                f"**Unknown vocabulary term** — `{v['subject']}` `{v['property']}` `{v['object']}` uses "
                f"`{v['term']}`, which its vocabulary does not define{hint}.",
                entities=[v["subject"], v["object"]] if v["object"].startswith("http") else [v["subject"]],
                triples=[(v["subject"], v["property"], v["object"])],
            ))
        for uri in integrity.get("untyped_individuals", []):
            out.append(Problem("untyped", f"**Untyped individual** `{uri}` — it has no class.", entities=[uri]))

        if self.fix_orphans:
            for o in orphans:
                out.append(Problem(
                    "orphan",
                    f"**Orphan {o['kind']}** `{o['uri']}` (\"{o['label']}\") — no relation to the rest. Connect it "
                    "only if the source text says explicitly what it is; otherwise leave it.",
                    entities=[o["uri"]],
                ))
        return out

    # ── Context gathering ─────────────────────────────────────────────────

    async def _entity(self, uri: str) -> dict | None:
        if uri not in self._entity_cache:
            self._entity_cache[uri] = await self._try_json("concept_get", {"uri": uri})
        return self._entity_cache[uri]

    async def _ancestors(self, uris: list[str]) -> dict[str, list[str]]:
        """Every class each entity is (transitively) a subclass or an instance of."""
        meta = ", ".join(f"<{t}>" for t in _META_TYPES)
        rows = await self._try_json("sparql_query", {"limit": 1000, "query": f"""
            PREFIX rdf: <{RDF_TYPE.rsplit('#', 1)[0]}#>
            PREFIX rdfs: <{RDFS}>
            SELECT DISTINCT ?e ?a WHERE {{ GRAPH <{self.graph}> {{
                VALUES ?e {{ {" ".join(f"<{u}>" for u in uris)} }}
                ?e (rdf:type|rdfs:subClassOf)/rdfs:subClassOf* ?a .
                FILTER(isIRI(?a) && ?a != ?e && ?a NOT IN ({meta}))
            }} }}"""})
        out: dict[str, list[str]] = {}
        for r in (rows or {}).get("rows", []):
            out.setdefault(r["e"], []).append(r["a"])
        return out

    async def _property_uses(self, prop: str, side: str) -> list[str]:
        var = "?s" if side == "subject" else "?o"
        rows = await self._try_json("sparql_query", {"limit": 15, "query": f"""
            SELECT DISTINCT {var} WHERE {{ GRAPH <{self.graph}> {{ ?s <{prop}> ?o . FILTER(isIRI(?o)) }} }}"""})
        return [r[var[1:]] for r in (rows or {}).get("rows", [])]

    async def _violation_hint(self, v: dict, kind: str, ancestors: dict[str, list[str]]) -> str:
        """What the LLM needs to choose between fixing the relation and widening the property."""
        side = "subject" if kind == "domain" else "object"
        offender, expected = v[side], v["expected"]
        shared = [a for a in [expected, *ancestors.get(expected, [])]
                  if a == offender or a in ancestors.get(offender, [])]
        uses = await self._property_uses(v["property"], side)
        lines = [
            f"  - Classes `{_local(offender)}` and `{_local(expected)}` have in common: "
            + (", ".join(f"`{a}`" for a in shared) if shared else "none (widening means clearing the "
               f"{kind} with an empty string)"),
            f"  - Current {side}s of `{_local(v['property'])}`: " + ", ".join(f"`{_local(u)}`" for u in uses),
        ]
        return "\n".join(lines)

    @staticmethod
    def _describe(uri: str, e: dict | None, ancestors: list[str]) -> str:
        if not e:
            return f"- `{uri}` — (not in this ontology: seed or external term)"
        by_pred: dict[str, list[str]] = {}
        for t in e.get("triples", []):
            by_pred.setdefault(t["predicate"], []).append(t["object"])
        parts = [f"- `{uri}`"]
        if label := e.get("label"):
            parts.append(f'"{label}"')
        if definition := e.get("definition"):
            parts.append("— " + (definition[:300] + "…" if len(definition) > 300 else definition))
        details = []
        for pred, name in ((RDF_TYPE, "type"), (RDFS + "subClassOf", "subClassOf"),
                           (RDFS + "domain", "domain"), (RDFS + "range", "range")):
            values = [_local(v) for v in by_pred.get(pred, []) if v and not v.startswith("_:")]
            if values:
                details.append(f"{name}: {', '.join(values)}")
        if ancestors:
            details.append(f"ancestors: {', '.join(_local(a) for a in ancestors)}")
        for r in e.get("restrictions", []):
            target = r.get("value") or ""
            card = f" {r['cardinality']}" if "cardinality" in r else ""
            details.append(f"restriction: {_local(r['property'])} {r['restriction_type']}{card} {_local(target)}".rstrip())
        return " ".join(parts) + (f" [{'; '.join(details)}]" if details else "")

    async def context(
        self, group: list[Problem], heading: str = "Problem", with_sources: bool = True,
    ) -> tuple[str, str]:
        """Problems text with the involved entities described, and the source chunk texts."""
        entities: list[str] = []
        for p in group:
            entities.extend(u for u in p.entities if u not in entities)
        entities = entities[:MAX_ENTITIES]

        # Sources: those of the axioms at stake first, then those of the entities.
        chunk_ids: list[str] = []
        if with_sources:
            for p in group:
                for s, prop, o in p.triples:
                    ids = await self._try_json(
                        "relation_sources", {"subject_uri": s, "property_uri": prop, "object_value": o},
                    )
                    chunk_ids.extend(i for i in ids or [] if i not in chunk_ids)
            for uri in entities:
                e = await self._entity(uri)
                chunk_ids.extend(i for i in (e or {}).get("source_chunk_ids", []) if i not in chunk_ids)
            chunk_ids = chunk_ids[:MAX_SOURCE_CHUNKS]

        ancestors = await self._ancestors(entities) if entities else {}
        blocks = []
        for i, p in enumerate(group, 1):
            text = f"### {heading} {i}\n{p.text}"
            if p.violation:
                text += "\n" + await self._violation_hint(p.violation, p.kind, ancestors)
            blocks.append(text)
        problems_text = "\n\n".join(blocks) + "\n\n## Entities involved\n" + "\n".join(
            [self._describe(u, await self._entity(u), ancestors.get(u, [])) for u in entities]
        )

        sources = "(no source chunk recorded)"
        if chunk_ids:
            chunks = await self._try_json("chunk_read_batch", {"chunk_ids": chunk_ids})
            if chunks is None:
                sources = "(source text unavailable)"
            else:
                texts = {c["id"]: re.sub(r"[ \t]+", " ", c["text"]) for c in chunks if c.get("text")}
                sources = "\n\n".join(
                    f"### Chunk {cid}\n" + (
                        text[:MAX_CHUNK_CHARS] + " …" if len(text) > MAX_CHUNK_CHARS else text
                    )
                    for cid, text in texts.items()
                )
        return problems_text, sources

    # ── Check & repair ────────────────────────────────────────────────────

    async def check(self) -> tuple[dict, list[dict]]:
        report = json.loads(await self._call("ontology_check", {"include_seeds": self.include_seeds}))
        if error := report.get("reasoner", {}).get("error"):
            self.log(f"Reasoner unavailable, only the SPARQL checks ran: {error}")
        orphans = json.loads(await self._call("ontology_orphans")) if self.fix_orphans else []
        return report, orphans

    def _record(self, label: str, report: dict, orphans: list[dict]) -> None:
        counts = report_counts(report, orphans)
        self.checks.append({"label": label, **counts})
        self.log(_format_counts(label, counts))

    async def _ensure_backup(self) -> None:
        """Turtle dump of the whole ontology graph (provenance included) before the first change."""
        if self.backup_cb and not self._backed_up:
            self.backup_cb(await self._call("ontology_export"))
            self._backed_up = True

    async def repair(self, label: str = "Round") -> bool:
        """Check → fix → check again, for up to max_rounds rounds. Returns whether the
        ontology is consistent at the end (unknown — no reasoner — counts as consistent)."""
        for round_number in range(1, self.max_rounds + 1):
            self._entity_cache.clear()
            report, orphans = await self.check()
            problems = await self.problems(report, orphans)
            self._record(f"{label} {round_number}", report, orphans)
            keys = {p.key for p in problems}
            if not problems:
                self.log("Nothing left to fix.")
                self._unresolved = keys
                break
            if keys == self._unresolved:
                self.log(f"Nothing changed since the last fixes — stopping with {len(problems)} problem(s) left.")
                break
            self._unresolved = keys
            await self._ensure_backup()

            groups = [problems[i : i + PROBLEMS_PER_TASK] for i in range(0, len(problems), PROBLEMS_PER_TASK)]
            for n, group in enumerate(groups, 1):
                problems_text, sources = await self.context(group)
                await self.converse(
                    FIX_PROMPT.format(count=len(group), problems=problems_text, sources=sources),
                    f"{label.lower()} {round_number} · task {n}/{len(groups)}",
                )
        else:
            self._entity_cache.clear()
            report, orphans = await self.check()
            self._record(f"{label} — final", report, orphans)
            self._unresolved = {p.key for p in await self.problems(report, orphans)}
        return report.get("reasoner", {}).get("consistent") is not False

    # ── Inferences ────────────────────────────────────────────────────────

    async def _preview_inferences(self) -> list[dict] | None:
        try:
            preview = json.loads(await self._call("ontology_infer", {
                "action": "preview", "include_seeds": self.include_seeds, "max_explained": MAX_EXPLAINED_INFERENCES,
            }))
        except RuntimeError as exc:
            self.log(f"Inference skipped: {exc}")
            return None
        items = preview["inferences"]
        self.log(
            f"Inferences: {len(items)} entailed but not asserted — {sum(i['taxonomic'] for i in items)} from the "
            f"class hierarchy alone, {sum(i['materialized'] for i in items)} already materialized by an earlier run."
        )
        return items

    async def _parent_counts(self, uris: list[str]) -> dict[str, int]:
        """Number of named superclasses each class has (asserted)."""
        if not uris:
            return {}
        rows = (await self._try_json("sparql_query", {"limit": 1000, "query": f"""
            PREFIX rdfs: <{RDFS}>
            SELECT ?c (COUNT(DISTINCT ?p) AS ?n) WHERE {{ GRAPH <{self.graph}> {{
                VALUES ?c {{ {" ".join(f"<{u}>" for u in uris)} }}
                ?c rdfs:subClassOf ?p . FILTER(isIRI(?p))
            }} }} GROUP BY ?c"""}) or {}).get("rows", [])
        return {r["c"]: int(r["n"] or 0) for r in rows}

    @staticmethod
    def _inference_problem(item: dict, parent_counts: dict[str, int]) -> Problem:
        s, p, o = item["subject"], item["predicate"], item["object"]
        names = item.get("entities", {})
        lines = []
        for axiom in (a for axioms in item["explanations"][:1] for a in axioms):
            note = ""
            m = re.fullmatch(r"(\S+) subClassOf (\S+)", axiom.strip())
            if m and isinstance(sub := names.get(m.group(1)), str) and parent_counts.get(sub) == 1:
                note = f" — the only parent of {m.group(1)}: deleting it leaves {m.group(1)} without any parent"
            lines.append(f"  - `{axiom}`{note}")
        why = "\n".join(lines)
        origin = "from the class hierarchy alone" if item["taxonomic"] else "involving domains, ranges, equivalences…"
        text = (f"**Inference** `{s}` `{p}` `{o}` ({origin})\n"
                + (f"Because of:\n{why}" if why else "(no explanation computed)"))
        entities = [s, o]
        for v in item.get("entities", {}).values():
            entities.extend(u for u in (v if isinstance(v, list) else [v]) if u not in entities)
        return Problem("inference", text, entities=entities)

    async def review(self, items: list[dict]) -> None:
        """Have the LLM read the new inferences: an absurd one reveals a wrong axiom in its
        explanation, which it fixes. Inferences with the same object are kept together, as
        they often come from the same axiom."""
        items = sorted(items, key=lambda i: (i["object"], i["predicate"], i["subject"]))
        groups = [items[i : i + INFERENCES_PER_TASK] for i in range(0, len(items), INFERENCES_PER_TASK)]
        for n, group in enumerate(groups, 1):
            self._entity_cache.clear()
            subjects = sorted({u for i in group for name, u in i.get("entities", {}).items()
                               if isinstance(u, str) and any(
                                   a.strip().startswith(f"{name} subClassOf ") for ax in i["explanations"][:1] for a in ax)})
            counts = await self._parent_counts(subjects)
            problems_text, sources = await self.context(
                [self._inference_problem(i, counts) for i in group], "Inference",
            )
            await self.converse(
                REVIEW_PROMPT.format(count=len(group), inferences=problems_text, sources=sources),
                f"review · task {n}/{len(groups)}",
            )

    async def infer_and_materialize(self) -> None:
        items = await self._preview_inferences()
        if items is None:
            return
        self.inferences = {"entailed": len(items), "taxonomic": sum(i["taxonomic"] for i in items)}
        to_review = [i for i in items if not i["materialized"]]
        if self.review_inferences and to_review:
            await self._ensure_backup()
            await self.review(to_review)
            # The review may have changed axioms: check again before materializing.
            if not await self.repair(label="After review"):
                self.log("The ontology is inconsistent after the review — inferences not materialized.")
                return
        await self._ensure_backup()
        result = json.loads(await self._call(
            "ontology_infer", {"action": "materialize", "include_seeds": self.include_seeds},
        ))
        self.inferences.update(materialized=result["stored"], replaced=result["replaced"])
        self.log(f"Inferences materialized: {result['stored']} (replacing {result['replaced']}).")

    # ── Deduplication ─────────────────────────────────────────────────────

    async def deduplicate(self) -> None:
        """Pairs of classes whose concept embeddings are closer than DEDUP_THRESHOLD are
        reviewed by the LLM, DEDUP_PAIRS_PER_TASK at a time: merge or keep."""
        classes = json.loads(await self._call("concept_list", {"limit": 100_000}))
        by_uri = {c["uri"]: c for c in classes}
        pairs: dict[tuple[str, str], float] = {}
        for c in classes:
            query = c["label"] + (f": {c['definition']}" if c.get("definition") else "")
            try:
                hits = json.loads(await self._call("concept_semantic_search", {"query": query, "top_k": 5}))
            except RuntimeError as exc:
                self.log(f"Deduplication skipped: {exc}")
                return
            for h in hits:
                if h["uri"] != c["uri"] and h["uri"] in by_uri and h["score"] >= DEDUP_THRESHOLD:
                    key = tuple(sorted((c["uri"], h["uri"])))
                    pairs[key] = max(pairs.get(key, 0.0), h["score"])

        self.dedup_pairs = len(pairs)
        if not pairs:
            self.log(f"Deduplication: no candidate pairs above {DEDUP_THRESHOLD:.2f}.")
            return
        self.log(f"Deduplication: {len(pairs)} candidate pairs above {DEDUP_THRESHOLD:.2f}.")
        await self._ensure_backup()

        def describe(uri: str) -> str:
            c = by_uri[uri]
            definition = (c.get("definition") or "")[:200]
            return f'"{c["label"]}" <{uri}>' + (f" — {definition}" if definition else "")

        ordered = sorted(pairs.items(), key=lambda kv: -kv[1])
        for start in range(0, len(ordered), DEDUP_PAIRS_PER_TASK):
            group = ordered[start : start + DEDUP_PAIRS_PER_TASK]
            text = "\n".join(f"- score {score:.3f}\n  A: {describe(a)}\n  B: {describe(b)}" for (a, b), score in group)
            await self.converse(
                DEDUP_PROMPT.format(pairs=text), f"dedup · task {start // DEDUP_PAIRS_PER_TASK + 1}",
            )

    # ── Enrichment: disjointness ──────────────────────────────────────────

    async def _sibling_groups(self) -> list[tuple[str, list[str]]]:
        """(parent, children) of the classes that share a parent (or are all roots) and have
        at least one pair neither disjoint nor in a subclass relation."""
        rows = (await self._try_json("sparql_query", {"limit": 1000, "query": f"""
            PREFIX owl: <{OWL}> PREFIX rdfs: <{RDFS}>
            SELECT ?c ?parent WHERE {{ GRAPH <{self.graph}> {{
                ?c a owl:Class . FILTER(isIRI(?c))
                OPTIONAL {{ ?c rdfs:subClassOf ?parent . FILTER(isIRI(?parent) && ?parent != owl:Thing && ?parent != ?c) }}
            }} }}"""}) or {}).get("rows", [])
        related = (await self._try_json("sparql_query", {"limit": 1000, "query": f"""
            PREFIX owl: <{OWL}> PREFIX rdfs: <{RDFS}>
            SELECT ?a ?b WHERE {{ GRAPH <{self.graph}> {{
                {{ ?a owl:disjointWith ?b }} UNION {{ ?a rdfs:subClassOf+ ?b . FILTER(isIRI(?b)) }}
            }} }}"""}) or {}).get("rows", [])
        excluded = {frozenset((r["a"], r["b"])) for r in related}

        children: dict[str, list[str]] = {}
        for r in rows:
            parent = r.get("parent") or "(root classes)"
            if r["c"] not in children.setdefault(parent, []):
                children[parent].append(r["c"])
        groups = []
        for parent, kids in sorted(children.items()):
            for start in range(0, len(kids), MAX_SIBLINGS_PER_GROUP):
                chunk = sorted(kids)[start : start + MAX_SIBLINGS_PER_GROUP]
                if any(frozenset((a, b)) not in excluded for i, a in enumerate(chunk) for b in chunk[i + 1:]):
                    groups.append((parent, chunk))
        return groups

    async def _disjoint_pairs(self) -> set[tuple[str, str]]:
        rows = (await self._try_json("sparql_query", {"limit": 1000, "query": f"""
            PREFIX owl: <{OWL}>
            SELECT ?a ?b WHERE {{ GRAPH <{self.graph}> {{ ?a owl:disjointWith ?b }} }}"""}) or {}).get("rows", [])
        return {(r["a"], r["b"]) for r in rows}

    async def enrich(self) -> bool:
        """Have the LLM declare disjoint sibling classes. Returns whether it was asked."""
        groups = [g for g in await self._sibling_groups() if len(g[1]) >= 2]
        if not groups:
            self.log("Enrichment: no sibling classes left to examine.")
            return False
        self.log(f"Enrichment: {len(groups)} groups of sibling classes to examine for disjointness.")
        await self._ensure_backup()
        before = await self._disjoint_pairs()
        per_task = SIBLING_GROUPS_PER_TASK
        for n, start in enumerate(range(0, len(groups), per_task), 1):
            self._entity_cache.clear()
            blocks = []
            for parent, kids in groups[start : start + per_task]:
                parent_name = parent if parent.startswith("(") else f"subclasses of `{parent}`"
                problem = Problem("siblings", f"Siblings — {parent_name}", entities=kids)
                text, _ = await self.context([problem], heading="Group", with_sources=False)
                blocks.append(text.replace("### Group 1", "###", 1))
            await self.converse(
                ENRICH_PROMPT.format(groups="\n\n".join(blocks)),
                f"enrich · task {n}/{-(-len(groups) // per_task)}",
            )
        added = await self._disjoint_pairs() - before
        self._added_disjoint |= added
        self.log(f"Enrichment: {len(added)} disjoint pairs declared.")
        return True

    # ── Run ───────────────────────────────────────────────────────────────

    async def run(self) -> dict:
        """Dedup → repair → enrich → infer. Returns what the run did: ``checks`` (the problem
        counts of every check, in order), ``changes`` (write tool calls), ``dedup_pairs``,
        ``disjoint_added`` and ``inferences`` (None when the step did not run)."""
        steps = ["repair"]
        if self.dedup:
            steps.insert(0, "dedup")
        if self.enrich_disjointness:
            steps.append("enrich")
        if self.infer:
            steps.append("infer")

        def step(name: str) -> None:
            self.set_progress(steps.index(name) + 1, len(steps))
            self.log(f"=== Step {steps.index(name) + 1}/{len(steps)}: {name} ===")

        if self.dedup:
            step("dedup")
            await self.deduplicate()
        step("repair")
        consistent = await self.repair()
        if self.enrich_disjointness:
            step("enrich")
            if not consistent:
                self.log("The ontology is still inconsistent — enrichment skipped.")
            elif await self.enrich():
                consistent = await self.repair(label="After enrichment")
        if self.infer:
            step("infer")
            if consistent:
                await self.infer_and_materialize()
            else:
                self.log("The ontology is still inconsistent — nothing can be inferred from it.")

        if self.export_cb and (self.changes or (self.inferences or {}).get("materialized")):
            content = await self._call("ontology_export", {"include_seeds": False})
            if has_ontology_content(content):
                self.export_cb(content)
        self.log(f"Reasoning done — {self.changes} change(s) applied.")
        return {
            "checks": self.checks,
            "changes": self.changes,
            "dedup_pairs": self.dedup_pairs,
            "disjoint_added": len(self._added_disjoint),
            "inferences": self.inferences,
        }


async def run_reasoning(session: ClientSession, **kwargs) -> dict:
    """Run :class:`ReasoningAgent` against an already-initialized OLAF session.

    ``kwargs`` are :class:`ReasoningAgent`'s keyword arguments. Returns what the
    run did (see :meth:`ReasoningAgent.run`).
    """
    tools_result = await session.list_tools()
    return await ReasoningAgent(session, tools_result.tools, **kwargs).run()
