package mastermind;

import java.util.Random;

import javafx.application.Application;
import javafx.scene.Scene;
import javafx.scene.layout.BorderPane;
import javafx.scene.layout.FlowPane;
import javafx.scene.layout.VBox;
import javafx.scene.control.*;
import javafx.scene.text.Text;
import javafx.stage.Stage;

public class MastermindGame extends Application 
{
    @Override
    public void start(Stage PRIMARY_STAGE) 
    {
        GameLogic logic = new GameLogic();
        GameGUI gui = new GameGUI(logic);
        gui.createGUI(PRIMARY_STAGE);
    }

    public static void main(String[] args) {
        launch(args);
    }
        
}






/* 

Pseudocode 
START

1. **Setup**
    - Import necessary libraries (Swing/JavaFX for GUI, Random for code generation, etc.).
    - Define constants for:
        - Code length (e.g., 4 digits/colors).
        - Number of attempts (e.g., 10).
        - Available colors/digits.
    - Initialize game variables:
        - Secret code (randomly generated).
        - User's current attempt count.

2. **Create GUI Components**
    - Main Window (JFrame/Stage):
        - Title: "Mastermind Game".
    - Input Panel:
        - Text fields or buttons for entering guesses.
        - Submit button.
    - Output Panel:
        - Feedback area for showing hints (correct position and correct color/digit).
        - Remaining attempts display.
    - Restart Button:
        - Resets the game.

3. **Generate Secret Code**
    - Randomly pick a combination of digits/colors based on the defined code length and options.
    - Store the code in an array or list.

4. **Add Event Listeners**
    - Attach event listener to the Submit button:
        - Retrieve the user's input.
        - Validate input length and format.
        - Compare input to the secret code:
            - Count correct digits/colors in the correct position (exact matches).
            - Count correct digits/colors in the wrong position (partial matches).
        - Update the feedback panel with the results.
        - Check win/loss condition:
            - If the user wins (exact match for all positions):
                - Display "Congratulations! You've cracked the code!".
                - Disable further input.
            - If the user loses (attempts exhausted):
                - Display "Game Over! The secret code was [code]".
                - Disable further input.
    - Attach event listener to the Restart button:
        - Reset all variables.
        - Clear input and feedback panels.
        - Generate a new secret code.

5. **Main Game Loop**
    - Display the GUI.
    - Wait for user interaction via Submit button.
    - Continue until:
        - User cracks the code.
        - User exhausts attempts.

6. **Helper Functions**
    - GenerateSecretCode():
        - Randomly generate and return a new secret code.
    - ValidateInput(String input):
        - Check if the input matches the required format and length.
    - CalculateFeedback(String input):
        - Compare user input with the secret code.
        - Return feedback as exact and partial matches.

7. **End Game**
    - Exit the application when the user chooses.

END
*/




/*
//import java.util.Random;
    private static final int CODE_LENGTH = 4;
    private static final int MAX_ATTEMPTS = 10;
    private static final String[] COLORS = {"RED", "GREEN", "YELLOW", "BLUE", "PURPLE", "WHITE"};
    private static int ATTEMPTS_LEFT = MAX_ATTEMPTS;
    private static String[] SECRET_CODE;
    
    // Constructor
    public Main(){
        ATTEMPTS_LEFT = MAX_ATTEMPTS;
        generateSecretCode();
    }

    // Generate a random secret code
    private void generateSecretCode(){
        Random random = new Random();
        SECRET_CODE = new String[CODE_LENGTH];
        for(int i = 0; i < CODE_LENGTH; i++){
            SECRET_CODE[i] = COLORS[random.nextInt(COLORS.length)];
        }
    }

    // Get the secret code (for testing or debugging)
    public String[] getSecretCode(){
        return SECRET_CODE;
    }

    // Get the remaining attempts
    public int getAttemptsleft(){
        return ATTEMPTS_LEFT;
    }

    // Reduce attempts by one
    public void reduceAttempt(){
        ATTEMPTS_LEFT--;
    }

    // Get feedback for the player's guess
    public String getFeedback(String[] PLAYER_GUESS){
        int BULL = 0;
        int COW = 0;

        boolean[] CODE_USED = new boolean[CODE_LENGTH];
        boolean[] GUESS_USED = new boolean[CODE_LENGTH];

        // check for Bull matches
        for(int i = 0; i < CODE_USED.length; i++){

            if(PLAYER_GUESS[i].equalsIgnoreCase(SECRET_CODE[i])){  
                BULL++; // 2

                CODE_USED[i] = true;
                GUESS_USED[i] = true;
            }


        }
        // check for Cow matches 
        for(int i = 0; i < CODE_LENGTH; i++){
            if(!GUESS_USED[i]){
                for(int j = 0; j <  CODE_LENGTH; j++){
                    if(!CODE_USED[i] && PLAYER_GUESS[j].equalsIgnoreCase(SECRET_CODE[j])){
                        COW++; 
                        CODE_USED[i] = true;
                        break;
                    }
                }
            }
        }
        
        return "BULL: " + BULL + "COW: " + COW + "\n";
    }

     // Check if the game is over
    public boolean isGameOver(){
        return ATTEMPTS_LEFT == 0;
    }

    // Check if the player has won
    public boolean isWin(String[] PLAYER_GUESS){
        for(int i = 0; i < CODE_LENGTH; i++){
            if(!PLAYER_GUESS[i].equalsIgnoreCase(SECRET_CODE[i])){

                return false;

            }
            
        }
        return true;

    }

    // Reset the game
    public void resetCode(){
        generateSecretCode();
        ATTEMPTS_LEFT = MAX_ATTEMPTS;
    }*/