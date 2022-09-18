# Reinforcement-Learning
IMBIZO PROJECT 2022

Exploring model based (dyna q) and model free reinforcement learning in a gridworld environment.
A model-based agent is an extension of the model-free agent. With the model-based agent, the value function takes an action and gains experience in the environment.
Experience gained is stored in the model and then it uses the model to sample so it can learn more of the the value function.
Created an obstacle gridworld to see if my learning agent (dyna q) could learn in this gridword. 
After my agent explored in this gridworld world, it was able to learn well despite the obstactles.
I then visualized the Q-values in the gridworld as the dyna q agent learns.
After seeing how the dyna q behaves, I then compared the two agents that's the model free and model-based agents to see which of them learns better in the environment.
From my result obtained, model-based agent learns better and faster than the model free agent.
I then visualized the Q-values for both model free and model-based agent as they learn.
With the planning steps, compared differnt planning steps to observe the step at which the agent performs better. With planning steps 2,5 and 10, the outcome showed that the more steps the agent takes the better it performs. So the agent turns to perform better at planning step 10.
