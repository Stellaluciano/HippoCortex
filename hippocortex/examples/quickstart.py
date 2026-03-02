from hippocortex.minimal import HippoCortexMemory


def main() -> None:
    memory = HippoCortexMemory()

    memory.add_event("user", "I prefer concise updates and metric units.")
    memory.add_event("assistant", "Acknowledged. I will default to concise metric responses.")
    memory.add_event("user", "I am planning a marathon training block for October.")

    notes = memory.consolidate()
    print("Consolidated notes:")
    for i, note in enumerate(notes, start=1):
        print(f"  {i}. {note.text} (from events {note.source_indices})")

    print("\nSearch: 'metric preferences and marathon context'")
    for hit in memory.search("metric preferences and marathon context", k=2):
        print(f"  - {hit.text}")


if __name__ == "__main__":
    main()
