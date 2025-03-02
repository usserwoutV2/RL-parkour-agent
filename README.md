
# Parkour AI Bot

This project is an AI-based bot designed to navigate and complete parkour maps in Minecraft. The bot uses reinforcement learning and genetic algorithms to improve its performance over time.

## Getting Started

### Prerequisites

- Python 3.x
- Node.js
- npm

### Installation

1. Clone the repository:

2. Install the necessary Python packages:
    ```sh
    pip install -r requirements.txt
    ```

3. Install the necessary Node.js packages:
    ```sh
    cd examples/minecraft
    npm install
    ```

### Running the Bot

To run the bot using the genetic algorithm, use the following command:
```sh
npm run bots
```

### Project Components

- **Bot.py**: Contains the main bot logic and interaction with the Minecraft environment.
- **geneticAlgo.py**: Implements the genetic algorithm for training the bot.
- **agent.py**: Contains the reinforcement learning agent and its training logic.
- **parkour.py**: Defines the parkour environment and related functions.

### Configuration

- **parkour_maps.json**: Contains the definitions of various parkour maps that the bot can attempt to complete.

