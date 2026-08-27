package software_project;

import org.junit.Test;
import static org.junit.Assert.*;

public class HomePageTest {
    @Test
    public void testShowCar() {
        HomePage instance = new HomePage(1);
        instance.showCar(1, "BMW", "2024", "$500");
        assertEquals(1, instance.selectedCarId);
        assertEquals("BMW", instance.selectedCar);
        assertEquals("BMW", instance.getNameCar().getText());
        assertEquals("2024", instance.getModelCar().getText());
        assertEquals("$500", instance.getPriceCar().getText());
        assertTrue(instance.getRentBTN().isEnabled());
        assertTrue(instance.getBuyBTN().isEnabled());
    }

    @Test
    public void testInvalidCar() {
        HomePage instance = new HomePage(1);
        instance.showCar(-1, "Choose a car", "-", "-");
        assertEquals(-1,instance.selectedCarId);
        assertEquals("Choose a car", instance.selectedCar);
    }

    @Test
    public void testMain() {
        String[] args = null;
        HomePage.main(args);
    }
}