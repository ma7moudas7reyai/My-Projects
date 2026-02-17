public class Main {

    private static final secretCodeLength = 4;
    private static final string[] colors = {"red", "blue", "green", "Purple", "black", "yellow"};
    private String[] secretCode;
    private static final maxAttmepts = 10;
    private static attemptLeft = maxAttmepts;

    // Setter to generate a random secret code 
    public static void setGenerateSecretCode(){
        Random random = new Random(); 
        secretCode = new String[secretCodeLength]; 
        boolean[] usedColors = new boolean[colors.length]; 
    
        for (int i = 0; i < secretCodeLength; i++) {
            int colorIndex; 
            do {
                colorIndex = random.nextInt(colors.length); 
            } while (usedColors[colorIndex]); 
    
            secretCode[i] = colors[colorIndex]; 
            usedColors[colorIndex] = true; 
        }
    }
}
// Getter to Display the secret code after game over
public static String[] getSecretCode(){
    return secretCode;
}

public static int getAttemptsLeft(){
    return attemptLeft;
}

public static void setReduceAttempt(){
    attemptLeft--;
}

public static String getFeedback(String[] userGuess){
    int bull = 0;
    int cow = 0;

    boolean[] codeUsed = new boolean[secretCodeLength];
    boolean[] guessUsed = new boolean[secretCodeLength];

    // Check for bull matches
    for(int i = 0; i < secretCodeLength; i++){
        if(userGuess[i].equalsIgnoreCase(secretCode[i])){ 
            bull++;
            codeUsed[i] = true;  // 0 = t, 1 = t  , 2 = f , 3 = t
            guessUsed[i] = true; // 0 = t, 1 = t  , 2 = f , 3 = t
        }
    }

    // Check for cow matches
    for(int i = 0; i < secretCodeLength; i++){
        if(!guessUsed[i]){
            for(int j = 0; j < secretCodeLength; j++){
                if(!codeUsed[j] && guessUsed[i].equalsIgnoreCase(secretCode[j])){ // r g p w    r g r w
                    cow++;
                    codeUsed[j] = true;
                    break;
                }
            }
        }
    }
    return "Bulls: " + bull + ", Cows: " + cow;
}

public static boolean isGameOver(){
    return attemptLeft <= 0;
}

public static boolean isWin(String[] userGuess){
    for(int i = 0; i < secretCodeLength; i++){
        if(!userGuess[i].equalsIgnoreCase(secretCode[i])){
            return false;
        }
    }
    return true;
}

public static void resetGame(){
    generateSecretCode();
    attemptLeft = maxAttmepts;
}

public static void main(String[] args){
    Scanner scanner = new Scanner(System.in);
    setGenerateSecretCode();
}



