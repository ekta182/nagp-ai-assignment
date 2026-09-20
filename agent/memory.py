"""Multi-turn context: a rolling transcript plus a durable preference store.

Two kinds of memory, because they behave differently. The transcript lets the
model resolve references like "there" or "the second day". The preference dict
persists across the whole session, so a budget mentioned in turn 2 is still
respected in turn 9 after it has scrolled out of the window.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from config import MAX_HISTORY_TURNS

PREFERENCE_LABELS = {
    "budget": "Budget",
    "trip_length_days": "Trip length (days)",
    "travellers": "Travelling with",
    "interests": "Interests",
    "dates": "Dates",
    "dietary": "Dietary needs",
    "pace": "Preferred pace",
    "notes": "Other notes",
}


@dataclass
class Turn:
    question: str
    answer: str


@dataclass
class ConversationMemory:
    turns: list[Turn] = field(default_factory=list)
    preferences: dict = field(default_factory=dict)

    def add_turn(self, question: str, answer: str) -> None:
        self.turns.append(Turn(question=question, answer=answer))

    def update_preferences(self, new: dict) -> dict:
        """Merge newly extracted preferences. Lists union, scalars overwrite."""
        changed = {}
        for key, value in (new or {}).items():
            if value in (None, "", [], {}, "null", "none", "unknown", "N/A"):
                continue
            if isinstance(value, list):
                existing = self.preferences.get(key) or []
                if not isinstance(existing, list):
                    existing = [existing]
                merged = list(dict.fromkeys([*existing, *value]))
                if merged != existing:
                    self.preferences[key] = merged
                    changed[key] = merged
            else:
                if self.preferences.get(key) != value:
                    self.preferences[key] = value
                    changed[key] = value
        return changed

    def history_text(self, max_turns: int | None = None, answer_chars: int = 700) -> str:
        max_turns = max_turns or MAX_HISTORY_TURNS
        recent = self.turns[-max_turns:]
        if not recent:
            return "(none)"
        parts = []
        for turn in recent:
            answer = turn.answer.strip()
            if len(answer) > answer_chars:
                answer = answer[:answer_chars].rsplit(" ", 1)[0] + " …"
            parts.append(f"User: {turn.question.strip()}\nAssistant: {answer}")
        return "\n\n".join(parts)

    def preferences_text(self) -> str:
        if not self.preferences:
            return "(none recorded yet)"
        parts = []
        for key, value in self.preferences.items():
            label = PREFERENCE_LABELS.get(key, key.replace("_", " ").title())
            if isinstance(value, list):
                value = ", ".join(str(v) for v in value)
            parts.append(f"{label}: {value}")
        return "; ".join(parts)

    def reset(self) -> None:
        self.turns.clear()
        self.preferences.clear()
