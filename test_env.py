from env import ABREnv


def main():
    env = ABREnv("context", split="train", seed=1)
    obs, _ = env.reset()
    assert obs.shape == (25,)
    done = False
    steps = 0
    while not done:
        obs, reward, done, _, info = env.step(0)
        assert obs.shape == (25,)
        assert "rebuffer" in info
        steps += 1
    print("PASS: ABR environment smoke tests")
    print(f"actions: {env.action_space.n}")
    print(f"chunks: {steps}")
    print(f"total rebuffer: {env.total_rebuffer:.3f}s")
    print(f"total reward: {env.total_reward:.3f}")


if __name__ == "__main__":
    main()
