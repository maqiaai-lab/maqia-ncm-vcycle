"""Sudoku helpers for the released NCM boards.

A puzzle is 91 ids: BOS, 89 board ids, BOS. A board is 9 rows of 9 cells separated by '|' (id 10),
so board position i holds row i // 10, column i % 10, and column 9 is a separator. Blank = 0.
"""
import os
import pickle

import torch

SEP, BLANK = 10, 0
CELL_POS = torch.tensor([r * 10 + c for r in range(9) for c in range(9)])
SEP_POS = torch.tensor([r * 10 + 9 for r in range(8)])


def _units():
    rows = [[r * 9 + c for c in range(9)] for r in range(9)]
    cols = [[r * 9 + c for r in range(9)] for c in range(9)]
    boxes = [[(br + r) * 9 + bc + c for r in range(3) for c in range(3)] for br in (0, 3, 6) for bc in (0, 3, 6)]
    return torch.tensor(rows + cols + boxes)  # (27, 9) indices into the 81 cells


UNITS = _units()


def load_split(data_dir, difficulty, n=None):
    d = os.path.join(data_dir, "sudoku", difficulty)
    puzzles = pickle.load(open(os.path.join(d, "puzzles.pickle"), "rb"))
    solutions = pickle.load(open(os.path.join(d, "solutions.pickle"), "rb"))
    n = n or len(puzzles)
    return torch.tensor(puzzles[:n]), torch.tensor(solutions[:n])


def clues(puzzles):
    """(mask, values) over the 89 board positions; mask is True on given cells."""
    board = puzzles[:, 1:90]
    return (board >= 1) & (board <= 9), board


def project(tokens, puzzles):
    """Pin the hard constraints we know at inference time: given clues and the '|' separators."""
    mask, vals = clues(puzzles)
    out = torch.where(mask.to(tokens.device), vals.to(tokens.device), tokens)
    out[:, SEP_POS.to(tokens.device)] = SEP
    return out


def grid(tokens):
    return tokens[:, CELL_POS.to(tokens.device)]  # (B, 81)


def unit_violations(tokens):
    """Per board: number of the 27 rows/columns/boxes that are not a permutation of 1..9."""
    g = grid(tokens)
    u = g[:, UNITS.to(tokens.device)]  # (B, 27, 9)
    onehot = F_one_hot(u)
    ok = (onehot.sum(2)[..., 1:10] == 1).all(-1)
    return (~ok).sum(-1)


def conflict_cells(tokens):
    """(B, 81) bool: cells that are not a digit or share a digit with another cell in some unit."""
    g = grid(tokens)
    units = UNITS.to(tokens.device)
    u = g[:, units]  # (B, 27, 9)
    dup = torch.gather(F_one_hot(u).sum(2), 2, u.clamp(0, 11)) > 1  # (B, 27, 9)
    bad = torch.zeros(g.shape[0], 81, dtype=torch.long, device=tokens.device)
    bad.index_add_(1, units.view(-1), dup.view(g.shape[0], -1).long())
    return (bad > 0) | (g < 1) | (g > 9)


def F_one_hot(u):
    return torch.nn.functional.one_hot(u.clamp(0, 11), 12)


def metrics(tokens, puzzles, solutions):
    """Per-board tensors: solved (exact match), cell accuracy, valid grid, unit violations, clue violations."""
    tokens, solutions = tokens.cpu(), solutions.cpu()
    g, gs = grid(tokens), grid(solutions)
    mask, vals = clues(puzzles.cpu())
    viol = unit_violations(tokens)
    digits_ok = ((g >= 1) & (g <= 9)).all(-1)
    return {
        "solved": (tokens == solutions).all(-1),
        "cell_acc": (g == gs).float().mean(-1),
        "valid": (viol == 0) & digits_ok,
        "unit_violations": viol,
        "clue_violations": ((tokens != vals) & mask).sum(-1),
    }
