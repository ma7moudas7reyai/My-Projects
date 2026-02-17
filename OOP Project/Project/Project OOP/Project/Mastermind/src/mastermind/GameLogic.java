package mastermind;

import java.util.Random;
public class GameLogic 
{
    static final int CODE_LENGTH = 4;
    private static final String[] COLORS = {"Red", "Black", "Green", "Yellow", "White", "Purple"};  // 4 0 1
    private String[] SECRET_CODE;
    private int ATTEMPTS_LEFT;

    public GameLogic() 
    {
        generateSecretCode();
        ATTEMPTS_LEFT = 10;     // Default number of attempts
    }

    public void generateSecretCode() 
    {
        Random random = new Random();                                   // red green yellow black
        SECRET_CODE = new String[CODE_LENGTH]; 
        boolean[] USED_COLORS = new boolean[COLORS.length];             
    
        for (int i = 0; i < CODE_LENGTH; i++) {
            int COLOR_INDEX; 
            do {
                COLOR_INDEX = random.nextInt(COLORS.length); 
            } while (USED_COLORS[COLOR_INDEX]); // used colors [4]  
    
            SECRET_CODE[i] = COLORS[COLOR_INDEX]; //white    red
            USED_COLORS[COLOR_INDEX] = true;  // 4 = true       0 = true  1 = ture 
        }
    }

    public String[] getSecretCode() 
    {
        return SECRET_CODE;
    }

    public int getAttemptsLeft() 
    {
        return ATTEMPTS_LEFT;
    }

    public void reduceAttempts() 
    {
        ATTEMPTS_LEFT--;
    }

    public String getFeedback(String[] PLAYER_GUESS) 
    {
        int BULL = 0;
        int COW = 0;

        boolean[] CODE_USED = new boolean[CODE_LENGTH];
        boolean[] GUESS_USED = new boolean[CODE_LENGTH];

        // Check for BULL matches
        for (int i = 0; i < CODE_LENGTH; i++)       //   red green blue black
        {                                           //   red green yellow black
            if (PLAYER_GUESS[i].equalsIgnoreCase(SECRET_CODE[i])) 
            {
                BULL++; // 2
                CODE_USED[i] = true;            // t t f f
                GUESS_USED[i] = true;           // t t f f 
            }
        }

        // Check for COW matches
        
        for (int i = 0; i < CODE_LENGTH; i++)   //   red green blue yellow
        {                  // 3                     //   red green yellow blue
            if (!GUESS_USED[i]) 
            {
                for (int j = 0; j < CODE_LENGTH; j++) 
                {
                    if (!CODE_USED[j] && PLAYER_GUESS[i].equalsIgnoreCase(SECRET_CODE[j])) 
                    {
                        COW++;
                        CODE_USED[j] = true;
                        break;
                    }
                }
            }
        }

        return "BULL: " + BULL  + ", COW: " + COW;
    }

    public boolean isGameOver() 
    {
        return ATTEMPTS_LEFT <= 0;
    }

    public boolean isWin(String[] PLAYER_GUESS) 
    {
        for (int i = 0; i < CODE_LENGTH; i++) 
        {
            if (!PLAYER_GUESS[i].equalsIgnoreCase(SECRET_CODE[i])) // red green blue yellow
            {                                                      // red green yellow blue                                                
                return false;
            }
        }
        return true;
    }

    public void resetGame() 
    {
        generateSecretCode();
        ATTEMPTS_LEFT = 10;
    };


}

