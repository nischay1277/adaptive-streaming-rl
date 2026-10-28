// Runs the REAL exported PPO actor in the browser (greedy action = argmax logits).
//
// Usage in simulator_web/index.html:
//   <script src="policy_infer.js"></script>
//   const policy = await loadPolicy('policy.json');
//   const action = policy.act(obs);   // obs must be built EXACTLY like env.py builds it
//
// Critical: the observation vector (order, scaling, history length) must match
// env.py / ppo_agent.py. If it does not, the policy will behave randomly.

async function loadPolicy(url) {
  const spec = await (await fetch(url)).json();
  const act = spec.activation === 'tanh' ? Math.tanh : (x) => Math.max(0, x);

  function act_(obs) {
    if (obs.length !== spec.input_dim) {
      throw new Error(`obs length ${obs.length} != expected ${spec.input_dim}`);
    }
    let x = obs.slice();
    if (spec.obs_mean && spec.obs_std) {
      x = x.map((v, i) => (v - spec.obs_mean[i]) / (spec.obs_std[i] + 1e-8));
    }
    spec.layers.forEach((L, li) => {
      const out = new Array(L.b.length);
      for (let r = 0; r < L.b.length; r++) {
        let s = L.b[r];
        const row = L.W[r];
        for (let c = 0; c < row.length; c++) s += row[c] * x[c];
        out[r] = li < spec.layers.length - 1 ? act(s) : s;
      }
      x = out;
    });
    let best = 0;
    for (let i = 1; i < x.length; i++) if (x[i] > x[best]) best = i;
    return best;
  }

  return { act: act_, spec };
}
