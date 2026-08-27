# Mastermind Game

A JavaFX implementation of the classic Mastermind code-breaking game, developed as an Object-Oriented Programming university project at the Egyptian E-Learning University (EELU).

## Demo

[Watch the Mastermind gameplay demo](demo/mastermind-demo.mp4)

## Features

- Generates a random four-color secret code with unique colors
- Accepts guesses through a graphical JavaFX interface
- Reports Bulls for correct colors in the correct positions
- Reports Cows for correct colors in different positions
- Tracks the remaining attempts
- Detects win and game-over states
- Supports restarting with a newly generated code

## Technologies and Concepts

- Java 17
- JavaFX
- Object-oriented programming
- Event-driven GUI development
- Separation of game logic and presentation

## Architecture

| Class | Responsibility |
| --- | --- |
| `MastermindGame` | Starts the JavaFX application |
| `GameGUI` | Builds the interface and handles user interaction |
| `GameLogic` | Generates codes, evaluates guesses, and manages game state |

## Run Locally

Requirements: JDK 17 and Maven.

```bash
mvn clean javafx:run
```

## Project Structure

```text
src/main/java/mastermind/
├── GameGUI.java
├── GameLogic.java
└── MastermindGame.java
```

Only the source code and reproducible build configuration are included. Compiled classes, IDE settings, and personal team documents are excluded.
