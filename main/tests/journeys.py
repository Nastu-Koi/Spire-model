"""A small world for the tests of the iteration: runs through several rooms
(`Journey`) and a policy checkpoint whose vocabulary covers them. No test of its own."""

import random

import torch

from main.network.config import ModelConfig, TrainConfig
from main.network.model import PolicyNetwork
from main.env.protocol import CHARACTERS, execution_command
from main.encoding.frames import clean_frame
from main.encoding.vocabulary import Vocabulary
from main.play.testing import SyntheticEngine, demonstration

class Journey(SyntheticEngine):
    """A run through several rooms: in each a fight (the selection, in the combat
    phase) and after it one decision outside the fight, standing on the room's floor.
    The decision after the last fight ends the run, won by playing the card; the
    earlier ones lead into the next room whatever is chosen. With acts, the last room
    of each act holds the act's boss fight."""

    def __init__(self, count=4, select=2, *, rooms=3, acts=None):
        """With `acts` (rooms per act) the rooms fall into acts of that many floors,
        the first room of an act on its first floor; without, every room is in act 3."""
        super().__init__(count, select)
        self.rooms, self.acts = rooms, acts

    def act_of(self, room):
        return room // self.acts + 1 if self.acts else 3

    def is_boss(self, room):
        return bool(self.acts) and (room + 1) % self.acts == 0

    def enter_room(self):
        """The fight of the current room: the act's boss in its last room."""
        self.enter(f"ordinary-{self.room}", self.act_of(self.room), "boss" if self.is_boss(self.room) else "normal")

    def reset(self, character="Ironclad", seed="test", ascension=10):
        self.room = 0
        frame = super().reset(character, seed, ascension)
        self.enter_room()
        return self.frame()

    def frame(self):
        frame = super().frame()
        if self.acts:
            frame["public"]["entities"][0].update(act=self.room // self.acts + 1, floor=self.room % self.acts + 1)
        else:
            frame["public"]["entities"][0].update(floor=self.room + 1)
        frame["public"]["entities"][0].update(gold=10 * self.room)
        if self.stage == "selection":
            frame["public"]["phase"] = "combat"
            # Each fight's selection is a selection of its own.
            frame["routing"]["selection_id"] += f":{self.room}"
        return frame

    def leave(self, candidate):
        if self.room + 1 == self.rooms:
            return super().leave(candidate)
        self.room += 1
        self.enter_room()

    def send(self, command):
        reply = super().send(command)
        if command["cmd"] == "write_continue_save":
            # The save names the room, which is all a fight entered again needs here.
            import json
            from pathlib import Path

            save = json.loads(Path(command["path"]).read_text())
            Path(command["path"]).write_text(json.dumps(dict(save, room=self.room, character=self.character)))
        return reply

    def resume(self, save):
        """Go on from a save written in a room: that room's fight again, from its first
        decision (`CliEngine.resume`)."""
        self.character = save.get("character", getattr(self, "character", "Ironclad"))
        self.seed, self.ascension = str(save["seed"]), getattr(self, "ascension", 0)
        self.commands, self.version, self.room = [], 0, save["room"]
        self.enter_room()
        return self.frame()

    def resume_fight(self, row):
        """Enter the fight of a saves row again (`CliEngine.resume_fight`): the room the
        save names, which must hold the recorded encounter's kind."""
        if row["kind"] != ("boss" if self.is_boss(row["save"]["room"]) else "normal"):
            raise ProtocolError("The save's room does not hold a fight of the recorded kind")
        return self.resume(row["save"])


TEMPERATURES = {"combat": 0.5, "other": 0.8}
ROOMS = 3


def engine():
    return Journey(4, 2, rooms=ROOMS)


def walked(character):
    """The frames of one run through the rooms, taking the first candidate throughout."""
    played = engine()
    frame, frames = played.reset(character, "vocabulary", 0), []
    while frame["boundary"] != "terminal":
        frames.append(clean_frame(frame))
        frame = played.send(execution_command(frame, frame["legal"]["candidates"][0]["candidate_ref"]))
    return frames


def learner(seed=0, **training):
    """A tiny policy whose vocabulary covers the rooms and the demonstrations."""
    from main.learn.trainer import Learner

    config = ModelConfig.tiny()
    frames = [frame for character in CHARACTERS for frame in walked(character)]
    frames += [step["frame"] for character in CHARACTERS
               for macro in demonstration(character, "vocabulary", 4, 2)["macros"] for step in macro["steps"]]
    settings = dict(optimizer="adamw", precision="no", loader_workers=0, logical_batch_size=8, microbatch_size=4,
                    bootstrap_warmup_updates=0, ascension=0, combat_temperature=TEMPERATURES["combat"],
                    other_temperature=TEMPERATURES["other"])
    torch.manual_seed(seed)
    return Learner(PolicyNetwork(config), Vocabulary.from_frames(frames, config), TrainConfig(**{**settings, **training}))


def checkpoint(path, **training):
    """A saved policy that resumes strictly, as the iterations start from one. The
    generators it is saved with are seeded: what resumes from it shuffles the same way."""
    policy = learner(**training)
    random.seed(0)
    policy.save(path)
    return path

