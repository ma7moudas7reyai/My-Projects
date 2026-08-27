package software_project;

import org.junit.Test;
import static org.junit.Assert.*;

public class LoginTest {
    @Test
    public void testValidEmail() {
        String email = "test@gmail.com";
        boolean expectedResult = true;
        boolean result = Login.isValidEmail(email);
        assertEquals(expectedResult, result);
    }

    @Test
    public void testEmailWithoutAt() {
        String email = "testgmail.com";
        boolean expectedResult = false;
        boolean result = Login.isValidEmail(email);
        assertEquals(expectedResult, result);
    }

    @Test
    public void testEmailWithoutName() {
        String email = "@gmail.com";
        boolean expectedResult = false;
        boolean result = SignUp.isValidEmail(email);
        assertEquals(expectedResult, result);
    }
    
    @Test
    public void testEmailWithoutDomain() {
        String email = "test@";
        boolean expectedResult = false;
        boolean result = Login.isValidEmail(email);
        assertEquals(expectedResult, result);
    }
    
    @Test
    public void testEmptyEmail() {
        String email = "";
        boolean expectedResult = false;
        boolean result = SignUp.isValidEmail(email);
        assertEquals(expectedResult, result);
    }

    @Test
    public void testValidPassword() {
        String password = "12345678";
        boolean expectedResult = true;
        boolean result = Login.isValidPassword(password);
        assertEquals(expectedResult, result);
    }

    @Test
    public void testValidPasswordWithLetters() {
        String password = "Ahmed123";
        boolean expectedResult = true;
        boolean result = SignUp.isValidPassword(password);
        assertEquals(expectedResult, result);
    }
    
    @Test
    public void testShortPassword() {
        String password = "123";
        boolean expectedResult = false;
        boolean result = Login.isValidPassword(password);
        assertEquals(expectedResult, result);
    }

    @Test
    public void testPasswordWithSpaces() {
        String password = "123 45678";
        boolean expectedResult = false;
        boolean result = Login.isValidPassword(password);
        assertEquals(expectedResult, result);
    }
    
    @Test
    public void testEmptyPassword() {
        String password = "";
        boolean expectedResult = false;
        boolean result = SignUp.isValidPassword(password);
        assertEquals(expectedResult, result);
    }
}