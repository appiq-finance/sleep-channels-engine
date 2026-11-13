# sleep-channels-engine

Code that makes and schedules the videos for two YouTube channels:

- **Stormwood Cabin** (@StormwoodCabin) — indoor cosy-cabin soundscapes: rain on the roof, fireplace, thunder, blizzards.
- **Valley & Tide** (@ValleyandTide) — outdoor nature sounds and weekly fictional sleep stories.

## How it works
1. `schedule/30_day_test.csv` lists every video: date, channel, title, length and sound recipe.
2. The **sound engine** (`engine/sound_engine.py`) builds each soundscape from code-generated layers (rain bed, roof patter, drops, fire crackles/pops/roar, thunder…). Every render is unique (seeded randomness), stereo is genuinely wide, loudness is mastered to a steady sleep-safe level with a true-peak ceiling.
3. The **Audio Quality Engineer** (`engine/quality_check.py`) measures every file: loudness, true peak, clipping, sudden jumps, silent dropouts, left/right balance. Anything that fails does not get uploaded.
4. GitHub Actions renders and uploads each approved video with a scheduled publish time; YouTube releases it on time.

No passwords, keys or channel access are stored in this code. They live only in GitHub's encrypted *Secrets*.
