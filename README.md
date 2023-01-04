# gym-atom-array

A 2-D array environment to facilitate application of Reinforcement Learning for the *Atom Rearrangement Problem*.

For background on the problem, and more details, refer to LINK.

## Environment details

For an array of size $N \times M$

### Observation Space

`Box` of shape (3, $N$, $M$)

0. Boolean array of atom positions
1. Boolean array of target sites
2. Tweezer position: 1 for empty tweezer, -1 for loaded tweezer

### Action Space

`Discrete` of size 6
- 0..3: Up, Down, Left, Right
- 4: Extract atom to tweezer
- 5: Release atom from tweezer

### Goal

Move atoms to fill the target sites. Extra atoms (not on the target sites) may remain, whose position does not matter.