import numpy as np
from scipy.linalg import block_diag

from env import GridWorld

np.set_printoptions(precision=4, suppress=True)


def Coin(angle):
    c = np.cos(2 * angle)
    s = np.sin(2 * angle)
    return np.array([[c, s], [s, -c]])


def block_diag_all(matrices):
    return block_diag(*matrices)


def ClementsLayer(dim, withSkip, angles):
    if withSkip == 0:
        blocks = [Coin(angles[k]) for k in range(dim // 2)]
        if dim % 2 == 1:
            blocks.append(np.array([[1.0]]))
        M = block_diag_all(blocks)
    else:
        blocks = [np.array([[1.0]])]
        for k in range((dim - 2) // 2 if dim % 2 == 0 else (dim - 1) // 2):
            composite = Coin(angles[k]) @ Coin(np.pi / 4)
            blocks.append(composite)
        if dim % 2 == 0:
            blocks.append(np.array([[1.0]]))
        M = block_diag_all(blocks)

        A_blocks = [Coin(np.pi / 4) for _ in range(dim // 2)]
        if dim % 2 == 1:
            blocks.append(np.array([[1.0]]))
        A = block_diag_all(A_blocks)
        M = A @ M

    return M


def ClementsMatrix(dim, Taskangles):
    M = np.eye(dim)
    for k in range(len(Taskangles)):
        M = ClementsLayer(dim, withSkip=k % 2, angles=Taskangles[k]) @ M
    return M


dim = 8

Taskangles = [
    [0.7746129631996155, 1.0149359703063965, 1.5167922973632812, 1.8602303266525269],
    [1.0792224407196045, 3.112532615661621, 1.5975351333618164],
    [1.8194468021392822, 2.9911093711853027, 1.679365634918213, 0.41973876953125],
    [1.748140573501587, 2.20228910446167, 0.28018873929977417],
    [1.462949514389038, 1.036205530166626, 0.4525461196899414, 2.9421119689941406],
    [0.05515575408935547, 1.4837419986724854, 1.3278334140777588],
    [0.21689969301223755, 1.7359626293182373, 2.8403542041778564, 1.2808785438537598],
    [0.5322084426879883, 2.7889726161956787, 1.7858834266662598],
]
M = ClementsMatrix(dim, Taskangles)
print(M)

num_percepts_list = [5, 3]
num_percepts = 15
num_actions = 4


def select_action(observation):
    print(f"Observation State: {observation}")
    input = np.zeros(dim, dtype=np.float32)
    offset = 0
    for i, val in enumerate(observation):
        input[offset + val] = 1 / np.sqrt(2)
        offset += num_percepts_list[i]
    print(f"Input Vector: {input}")
    output = (M @ input)[:num_actions]
    intensity = output**2
    probs = intensity / intensity.sum()
    print(f"Action Probs: {probs}")
    return int(np.random.choice(len(probs), p=probs))


dimensions = (5, 3)
invalid_cells = {(1, 1), (3, 0)}
rewards = {
    (4, 0): 10,
}
start_position = (0, 0)

env = GridWorld(
    dimensions=dimensions,
    start_position=start_position,
    invalid_cells=invalid_cells,
    rewards=rewards,
)
observation = env.reset()
print("Start at:", observation)

for step in range(10):
    action = select_action(observation)
    env.render()
    next_observation, reward, done = env.move(action)

    print(
        f"Step {step}: Action {env.act_names[action]}, New State {next_observation}, Reward {reward}, Done {done}"
    )
    observation = next_observation
    if done:
        print("Episode finished, resetting.")

# for x in range(5):
#     for y in range(3):
#         action = select_action([x,y])
#         print(action)