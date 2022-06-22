"""
Script for fine-grained grounding experiments; simulate natural interactions between
agent (learner) and user (teacher) with varying configurations
"""
import os
import sys
sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

import tqdm

from itl import ITLAgent
from itl.opts import parse_arguments
from tools.sim_user import SimulatedTeacher


TAB = "\t"

if __name__ == "__main__":
    opts = parse_arguments()
    agent = ITLAgent(opts)
    user = SimulatedTeacher(opts, target_concepts=["brandy_glass.n.*", "burgundy_glass.n.*"])

    for i in tqdm.tqdm(range(100), total=100):
        print("")
        print(f"Sys> ITL episode {i+1}")
        # Each single ITL episode is initiated by the teacher, aiming to test and confer
        # knowledge on one of the target concepts specified
        user_init = user.initiate_episode()
        agent_reaction = agent.loop(**user_init)

        # Continue interaction until agent answers with "OK."
        while "OK." not in agent_reaction:
            user_reaction = user.react(agent_reaction)
            agent_reaction = agent.loop(**user_reaction)
        
        # End of episode, push record to history
        user.episode_records.append(user.current_record)

    with open("result.csv", "w") as out_csv:
        # Summarize ITL interaction records stored in the simulated user object
        print("")
        print("Sys> Experiment finished. Result:")

        cumul = 0
        for i, ep in enumerate(user.episode_records):
            if ep["answer_correct"]:
                cumul += 1
                print(f"Sys> {TAB}Episode {i+1}: Correct")
            else:
                answer = ep["answered_concept"]
                ground_truth = ep["target_concept"]
                print(f"Sys> {TAB}Episode {i+1}: Wrong")
                print(f"Sys> {TAB*2}Answered: {answer} vs. Correct: {ground_truth}")

            out_csv.write(f"{i},{cumul}\n")
