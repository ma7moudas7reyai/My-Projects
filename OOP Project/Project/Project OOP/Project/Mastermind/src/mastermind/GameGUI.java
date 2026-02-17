package mastermind;

import javafx.scene.Scene;
import javafx.scene.control.*;
import javafx.scene.layout.*;
import javafx.stage.Stage;
import javafx.geometry.Pos;
import javafx.scene.paint.Color;
import javafx.scene.text.Font;

public class GameGUI {
    private GameLogic gameLogic;
    private TextField[] guessFields;
    private TextArea feedbackArea;
    private Button submitButton, restartButton;

    public GameGUI(GameLogic logic) {
        this.gameLogic = logic;
    }

    public void createGUI(Stage stage) {
        // Set up the root layout with a dark background color
        BorderPane root = new BorderPane();
        root.setStyle("-fx-background-color: #263238;"); // Dark greyish-blue background for a sleek, modern look

        // Input Panel (for entering guesses)
        FlowPane inputPanel = new FlowPane();
        inputPanel.setHgap(10);
        inputPanel.setVgap(10);
        inputPanel.setAlignment(Pos.CENTER);

        guessFields = new TextField[GameLogic.CODE_LENGTH];
        for (int i = 0; i < GameLogic.CODE_LENGTH; i++) {
            guessFields[i] = new TextField();
            guessFields[i].setPromptText("Color " + (i + 1));
            guessFields[i].setPrefWidth(90);
            guessFields[i].setStyle("-fx-background-color: #455a64; -fx-text-fill: #ffffff; -fx-border-color: #37474f;");
            inputPanel.getChildren().add(guessFields[i]);
        }

        // Submit button styling with rounded corners and a dark green color
        submitButton = new Button("Submit");
        submitButton.setStyle("-fx-background-color: #388e3c; -fx-text-fill: white; -fx-font-size: 14px; -fx-padding: 10 20; -fx-border-radius: 5px;");
        submitButton.setPrefWidth(120);
        inputPanel.getChildren().add(submitButton);
        
        root.setTop(inputPanel);

        // Output Panel (for showing feedback)
        VBox outputPanel = new VBox(10);
        outputPanel.setAlignment(Pos.CENTER);

        // Feedback area with white background, black text, and improved spacing
        feedbackArea = new TextArea();
        feedbackArea.setEditable(false);
        feedbackArea.setStyle("-fx-background-color: #37474f; -fx-text-fill: #000000; -fx-font-size: 14px; -fx-padding: 10px; -fx-border-color: #388e3c;");
        feedbackArea.setPrefHeight(200); // Increased height for feedback area
        feedbackArea.setPrefWidth(350);  // Adjusted width for better layout
        feedbackArea.setWrapText(true);  // Ensures the text wraps and doesn't overflow
        outputPanel.getChildren().add(feedbackArea);

        // Restart button with a distinct color
        restartButton = new Button("Restart");
        restartButton.setStyle("-fx-background-color: #f44336; -fx-text-fill: white; -fx-font-size: 14px; -fx-padding: 10 20; -fx-border-radius: 5px;");
        restartButton.setPrefWidth(120);
        outputPanel.getChildren().add(restartButton);

        root.setCenter(outputPanel);

        // Add Listeners for buttons
        addListeners();

        // Scene Setup with increased dimensions
        Scene scene = new Scene(root, 500, 600); // Increased width and height for a more spacious layout
        stage.setTitle("Mastermind Game");
        stage.setScene(scene);
        stage.show();
    }

    private void addListeners() {
        submitButton.setOnAction(e -> {
            String[] userGuess = new String[GameLogic.CODE_LENGTH];
            for (int i = 0; i < GameLogic.CODE_LENGTH; i++) {
                userGuess[i] = guessFields[i].getText().trim();
            }

            if (gameLogic.isWin(userGuess)) {
                feedbackArea.appendText("Congratulations! You've cracked the code!\n");
                disableInputs();
            } else {
                String feedback = gameLogic.getFeedback(userGuess);
                gameLogic.reduceAttempts();
                feedbackArea.appendText("Guess: " + String.join(", ", userGuess) + " | " + feedback + 
                    " | Attempts Left: " + gameLogic.getAttemptsLeft() + "\n");
                if (gameLogic.isGameOver()) {
                    feedbackArea.appendText("Game Over! The secret code was " + String.join(", ", gameLogic.getSecretCode()) + "\n");
                    disableInputs();
                }
            }
        });

        restartButton.setOnAction(e -> {
            gameLogic.resetGame();
            feedbackArea.clear();
            for (TextField field : guessFields) {
                field.clear();
            }
            enableInputs();
        });
    }

    private void disableInputs() {
        for (TextField field : guessFields) {
            field.setDisable(true);
        }
        submitButton.setDisable(true);
    }

    private void enableInputs() {
        for (TextField field : guessFields) {
            field.setDisable(false);
        }
        submitButton.setDisable(false);
    }
}











/* 
package mastermind;

// import javafx.application.Platform;
import javafx.scene.Scene;
import javafx.scene.control.*;
import javafx.scene.layout.*;
// import javafx.scene.text.Text;
import javafx.stage.Stage;

public class GameGUI {
    private GameLogic gameLogic;
    private TextField[] guessFields;
    private TextArea feedbackArea;
    private Button submitButton, restartButton;

    public GameGUI(GameLogic logic) {
        this.gameLogic = logic;
    }

    public void createGUI(Stage stage) {
        BorderPane root = new BorderPane();

        // Input Panel
        FlowPane inputPanel = new FlowPane();
        guessFields = new TextField[GameLogic.CODE_LENGTH];
        for (int i = 0; i < GameLogic.CODE_LENGTH; i++) {
            guessFields[i] = new TextField();
            guessFields[i].setPromptText("Color " + (i + 1));
            inputPanel.getChildren().add(guessFields[i]);
        }
        submitButton = new Button("Submit");
        inputPanel.getChildren().add(submitButton);
        root.setTop(inputPanel);

        // Output Panel
        VBox outputPanel = new VBox();
        feedbackArea = new TextArea();
        feedbackArea.setEditable(false);
        outputPanel.getChildren().add(feedbackArea);

        restartButton = new Button("Restart");
        outputPanel.getChildren().add(restartButton);
        root.setCenter(outputPanel);

        // Add Listeners
        addListeners();

        // Scene Setup
        Scene scene = new Scene(root, 400, 600);
        stage.setTitle("Mastermind Game");
        stage.setScene(scene);
        stage.show();
    }

    private void addListeners() {
        submitButton.setOnAction(e -> {
            String[] userGuess = new String[GameLogic.CODE_LENGTH];
            for (int i = 0; i < GameLogic.CODE_LENGTH; i++) {
                userGuess[i] = guessFields[i].getText().trim();
            }

            if (gameLogic.isWin(userGuess)) {
                feedbackArea.appendText("Congratulations! You've cracked the code!\n");
                disableInputs();
            } else {
                String feedback = gameLogic.getFeedback(userGuess);
                gameLogic.reduceAttempts();
                feedbackArea.appendText("Guess: " + String.join(", ", userGuess) + " | " + feedback + 
                    " | Attempts Left: " + gameLogic.getAttemptsLeft() + "\n");
                if (gameLogic.isGameOver()) {
                    feedbackArea.appendText("Game Over! The secret code was " + String.join(", ", gameLogic.getSecretCode()) + "\n");
                    disableInputs();
                }
            }
        });

        restartButton.setOnAction(e -> {
            gameLogic.resetGame();
            feedbackArea.clear();
            for (TextField field : guessFields) {
                field.clear();
            }
            enableInputs();
        });
    }

    private void disableInputs() {
        for (TextField field : guessFields) {
            field.setDisable(true);
        }
        submitButton.setDisable(true);
    }

    private void enableInputs() {
        for (TextField field : guessFields) {
            field.setDisable(false);
        }
        submitButton.setDisable(false);
    }
}

*/