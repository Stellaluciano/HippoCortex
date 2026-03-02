from __future__ import annotations

import argparse

from hippocortex import HippoCortex


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m hippocortex", description="HippoCortex CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("init", help="Initialize local HippoCortex DB")

    add = sub.add_parser("add-event", help="Add event to episodic store")
    add.add_argument("--agent-id", required=True)
    add.add_argument("--session-id", required=True)
    add.add_argument("--role", required=True)
    add.add_argument("--content", required=True)

    con = sub.add_parser("consolidate", help="Run replay consolidation")
    con.add_argument("--agent-id", required=True)
    con.add_argument("--session-id")

    search = sub.add_parser("search", help="Search semantic notes")
    search.add_argument("--agent-id", required=True)
    search.add_argument("--query", required=True)
    search.add_argument("--k", type=int, default=5)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    memory = HippoCortex.default()

    if args.command == "init":
        print(f"Initialized HippoCortex with DB at {memory.config.db_path}")
    elif args.command == "add-event":
        event = memory.hippo.add_event(
            agent_id=args.agent_id,
            session_id=args.session_id,
            role=args.role,
            content=args.content,
            metadata={},
        )
        print(f"Added event id={event.id}")
    elif args.command == "consolidate":
        output = memory.consolidate(agent_id=args.agent_id, session_id=args.session_id)
        print(f"Consolidated {output.notes_created} notes from {len(output.episode_ids)} episodes")
    elif args.command == "search":
        results = memory.cortex.search(agent_id=args.agent_id, query=args.query, k=args.k)
        for hit in results:
            print(f"{hit.score:.3f}\t{hit.note.text}")
    else:
        parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
