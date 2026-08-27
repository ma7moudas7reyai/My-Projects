package software_project;

import org.junit.Test;
import static org.junit.Assert.*;

public class AdminsTest {
    @Test
    public void testLoadCars() {
        Admins instance = new Admins(1);
        instance.loadCars();
        int carsCount = instance.getCarsBox().getItemCount();
        assertTrue(carsCount > 0);
    }
    
    @Test
    public void testLoadTransactions() {
        Admins instance = new Admins(1);
        instance.loadTransactions();
        int componentsCount =  instance.getTransactionNav().getComponentCount();
        assertTrue(componentsCount > 0);
    }

    @Test
    public void testMain() {
        String[] args = null;
        Admins.main(args);
    }
}