"""Independent pure-Python implementations of xoshiro256++ and splitmix64, written
from the published algorithm descriptions (Blackman & Vigna), used to cross-check
the reference vectors embedded in the Rust tests."""
M = (1 << 64) - 1


def rotl(x, k):
    return ((x << k) | (x >> (64 - k))) & M


def xoshiro256pp(state, n):
    s = list(state)
    out = []
    for _ in range(n):
        result = (rotl((s[0] + s[3]) & M, 23) + s[0]) & M
        t = (s[1] << 17) & M
        s[2] ^= s[0]
        s[3] ^= s[1]
        s[1] ^= s[2]
        s[0] ^= s[3]
        s[2] ^= t
        s[3] = rotl(s[3], 45)
        out.append(result)
    return out


def splitmix64(seed, n):
    x = seed
    out = []
    for _ in range(n):
        x = (x + 0x9E3779B97F4A7C15) & M
        z = x
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & M
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & M
        out.append(z ^ (z >> 31))
    return out


print("xoshiro256++ [1,2,3,4]:", xoshiro256pp([1, 2, 3, 4], 10))
print("splitmix64 1234567:", splitmix64(1234567, 5))

recalled_x = [41943041, 58720359, 3588806011781223, 3591011842654386, 9228616714210784205,
              9973669472204895162, 14011001112246962877, 12406186145184390807,
              15849039046786891736, 10450023813501588000]
recalled_s = [6457827717110365317, 3203168211198807973, 9817491932198370423,
              4593380528125082431, 16408922859458223821]
print("xoshiro matches recalled published vector:", xoshiro256pp([1, 2, 3, 4], 10) == recalled_x)
print("splitmix matches recalled published vector:", splitmix64(1234567, 5) == recalled_s)

# open-interval mapping check
u_max = ((((M >> 12)) + 0.5) * 2.0 ** -52)
u_min = (0 + 0.5) * 2.0 ** -52
print("mapping at raw=MAX:", repr(u_max), "< 1:", u_max < 1.0, " raw=0:", repr(u_min), "> 0:", u_min > 0)
old_max = ((M >> 11) + 0.5) * 2.0 ** -53
print("old mapping at raw=MAX:", repr(old_max))
