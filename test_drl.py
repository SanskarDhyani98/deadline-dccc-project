from drl.dqn_agent import DQNAgent

agent = DQNAgent(
    state_size=7,
    action_size=5
)

state = [0.8, 0.5, 1, 0.2, 0.3, 0.4, 0.0]

action = agent.act(state)

print("Selected action:", action)

agent.remember(
    state,
    action,
    reward=1.0,
    next_state=state,
    done=False
)

loss = agent.train()

print("Training loss:", loss)
print("DRL test completed")