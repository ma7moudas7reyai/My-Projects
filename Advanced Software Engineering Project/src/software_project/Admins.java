package software_project;

import java.awt.BorderLayout;
import java.awt.Dimension;
import java.sql.*;
import javax.swing.*;

public class Admins extends javax.swing.JFrame {    
    int currentAdminId;
    
    private static final java.util.logging.Logger logger = java.util.logging.Logger.getLogger(Admins.class.getName());

    public Admins() {
        initComponents();
        transactionNav.removeAll();
        scrollPanel.getVerticalScrollBar().setUnitIncrement(16);
        
        addCarPanel.setVisible(false);
        deleteCarPanel.setVisible(false);
        transactionNav.setLayout(new BoxLayout(transactionNav, BoxLayout.Y_AXIS));

        loadCars();
        loadTransactions();
        addBTN.addActionListener(this::addBTNActionPerformed);
        deleteBTN.addActionListener(this::deleteBTNActionPerformed);
    }
    
    public Admins(int adminId) {
        initComponents();
        transactionNav.removeAll();
        this.currentAdminId = adminId;
        
        addBTN.addActionListener(this::addBTNActionPerformed);
        deleteBTN.addActionListener(this::deleteBTNActionPerformed);

        addCarPanel.setVisible(false);
        deleteCarPanel.setVisible(false);

        transactionNav.setLayout(new BoxLayout(transactionNav, BoxLayout.Y_AXIS));

        loadCars();
        loadTransactions();
    }
    
    void loadCars() {
        try (
            Connection con = DBConnection.connect();
            PreparedStatement pst = con.prepareStatement("SELECT id, name FROM Cars WHERE status = 'available'");
            ResultSet rs = pst.executeQuery()) {

            carsBox.removeAllItems();

            while (rs.next()) {
               carsBox.addItem(rs.getInt("id") + " - " + rs.getString("name"));
            }
        } catch (Exception e) {
           e.printStackTrace();
        }
    }
    
    private void addBTNActionPerformed(java.awt.event.ActionEvent evt) {
        String name = nameTextField.getText();
        String model = modelTextField.getText();
        String price = priceTextField.getText();

        if(name.isEmpty() || model.isEmpty() || price.isEmpty()) {
            JOptionPane.showMessageDialog(this, "Fill all fields!");
            return;
        }

        try {
            Connection con = DBConnection.connect();

            String query = "INSERT INTO Cars (name, model, price, status) VALUES (?, ?, ?, 'available')";
            PreparedStatement pst = con.prepareStatement(query, Statement.RETURN_GENERATED_KEYS);

            pst.setString(1, name);
            pst.setString(2, model);
            pst.setDouble(3, Double.parseDouble(price));

            pst.executeUpdate();

            ResultSet rs = pst.getGeneratedKeys();
            int carId = -1;

            if (rs.next()) {
                carId = rs.getInt(1);
            }
           
            String trans = "INSERT INTO Transactions (admin_id, car_id, type) VALUES (?, ?, 'add')";
            PreparedStatement pst2 = con.prepareStatement(trans);

            pst2.setInt(1, currentAdminId);
            pst2.setInt(2, carId);
            pst2.executeUpdate();

            JOptionPane.showMessageDialog(this, "Car Added!");

            loadCars();
            loadTransactions();

        } catch (Exception e) {
            e.printStackTrace();
        }
        
        nameTextField.setText("");
        modelTextField.setText("");
        priceTextField.setText("");
    }
    
    private void deleteBTNActionPerformed(java.awt.event.ActionEvent evt) {
        String selectedCar = (String) carsBox.getSelectedItem();

        if(selectedCar == null) {
            JOptionPane.showMessageDialog(this, "Select a car first!");
            return;
        }

        try (Connection con = DBConnection.connect()) {
 
            int carId = Integer.parseInt(selectedCar.split(" - ")[0]);

            String delete = "DELETE FROM Cars";
            PreparedStatement pst2 = con.prepareStatement(delete);
            pst2.setInt(1, carId);
            pst2.executeUpdate();

            String trans = "INSERT INTO Transactions (admin_id, car_id, type) VALUES (?, ?, 'delete')";
            PreparedStatement pst3 = con.prepareStatement(trans);
            pst3.setInt(1, currentAdminId);
            pst3.setInt(2, carId);
            pst3.executeUpdate();

            JOptionPane.showMessageDialog(this, "Car Deleted!");

            loadCars();
            loadTransactions();

        } catch (Exception e) {
            e.printStackTrace();
        }
    }
    
    void loadTransactions() {
        transactionNav.removeAll();

        try {
            Connection con = DBConnection.connect();

            String query = """
                SELECT 
                    c.name AS car_name,
                    t.type,
                    CASE 
                        WHEN t.user_id IS NOT NULL THEN u.name
                        ELSE a.name
                    END AS done_by,
                    CASE 
                        WHEN t.user_id IS NOT NULL THEN 'User'
                        ELSE 'Admin'
                    END AS role,
                    t.date
                FROM Transactions t
                LEFT JOIN Cars c ON t.car_id = c.id
                LEFT JOIN Users u ON t.user_id = u.id
                LEFT JOIN Users a ON t.admin_id = a.id
                ORDER BY t.date DESC
            """;

            PreparedStatement pst = con.prepareStatement(query);
            ResultSet rs = pst.executeQuery();

            while (rs.next()) {

                String text = rs.getString("car_name") + " was " + rs.getString("type") + " by " + rs.getString("done_by");

                JPanel item = new JPanel();
                item.setBackground(new java.awt.Color(30,41,59));
                item.setPreferredSize(new java.awt.Dimension(250,60));
                item.setMaximumSize(new java.awt.Dimension(250,60));
                item.setLayout(new java.awt.BorderLayout());
                item.setBorder(BorderFactory.createEmptyBorder(10,10,10,10));

                JLabel label = new JLabel(text);
                label.setForeground(new java.awt.Color(255,255,255));

                item.add(label, BorderLayout.CENTER);

                transactionNav.add(item);
                transactionNav.add(Box.createVerticalStrut(10));
            }

            transactionNav.revalidate();
            transactionNav.repaint();

        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    @SuppressWarnings("unchecked")
    // <editor-fold defaultstate="collapsed" desc="Generated Code">//GEN-BEGIN:initComponents
    private void initComponents() {

        jPanel1 = new javax.swing.JPanel();
        leftSideAdmin = new javax.swing.JPanel();
        addCar = new javax.swing.JLabel();
        addCarPanel = new javax.swing.JPanel();
        nameLabel = new javax.swing.JLabel();
        nameTextField = new javax.swing.JTextField();
        modelText = new javax.swing.JLabel();
        modelTextField = new javax.swing.JTextField();
        priceText = new javax.swing.JLabel();
        priceTextField = new javax.swing.JTextField();
        addBTN = new javax.swing.JButton();
        middleSideAdmin = new javax.swing.JPanel();
        deleteCAr = new javax.swing.JLabel();
        deleteCarPanel = new javax.swing.JPanel();
        carsBox = new javax.swing.JComboBox<>();
        deleteBTN = new javax.swing.JButton();
        rightSideAdmin = new javax.swing.JPanel();
        viewTransaction = new javax.swing.JLabel();
        viewTransactionLabel = new javax.swing.JPanel();
        scrollPanel = new javax.swing.JScrollPane();
        transactionNav = new javax.swing.JPanel();
        item1 = new javax.swing.JPanel();

        setDefaultCloseOperation(javax.swing.WindowConstants.EXIT_ON_CLOSE);

        jPanel1.setBackground(new java.awt.Color(15, 23, 42));
        jPanel1.setPreferredSize(new java.awt.Dimension(800, 500));

        leftSideAdmin.setBackground(new java.awt.Color(30, 41, 59));
        leftSideAdmin.setPreferredSize(new java.awt.Dimension(265, 500));

        addCar.setFont(new java.awt.Font("Segoe UI", 1, 18)); // NOI18N
        addCar.setForeground(new java.awt.Color(255, 255, 255));
        addCar.setText("                    Add Car");
        addCar.setBorder(javax.swing.BorderFactory.createLineBorder(new java.awt.Color(0, 0, 0)));
        addCar.setPreferredSize(new java.awt.Dimension(265, 40));
        addCar.addMouseListener(new java.awt.event.MouseAdapter() {
            public void mouseClicked(java.awt.event.MouseEvent evt) {
                addCarMouseClicked(evt);
            }
        });

        addCarPanel.setBorder(javax.swing.BorderFactory.createLineBorder(new java.awt.Color(0, 0, 0)));

        nameLabel.setFont(new java.awt.Font("Segoe UI", 1, 14)); // NOI18N
        nameLabel.setText("Name:");

        nameTextField.setBackground(new java.awt.Color(15, 23, 42));
        nameTextField.setForeground(new java.awt.Color(255, 255, 255));
        nameTextField.setBorder(javax.swing.BorderFactory.createLineBorder(new java.awt.Color(0, 0, 0)));

        modelText.setFont(new java.awt.Font("Segoe UI", 1, 14)); // NOI18N
        modelText.setText("Model:");

        modelTextField.setBackground(new java.awt.Color(15, 23, 42));
        modelTextField.setForeground(new java.awt.Color(255, 255, 255));
        modelTextField.setBorder(javax.swing.BorderFactory.createLineBorder(new java.awt.Color(0, 0, 0)));

        priceText.setFont(new java.awt.Font("Segoe UI", 1, 14)); // NOI18N
        priceText.setText("Price:");

        priceTextField.setBackground(new java.awt.Color(15, 23, 42));
        priceTextField.setForeground(new java.awt.Color(255, 255, 255));
        priceTextField.setBorder(javax.swing.BorderFactory.createLineBorder(new java.awt.Color(0, 0, 0)));

        addBTN.setBackground(new java.awt.Color(225, 29, 72));
        addBTN.setForeground(new java.awt.Color(255, 255, 255));
        addBTN.setText("Add");
        addBTN.addActionListener(this::addBTNActionPerformed);

        javax.swing.GroupLayout addCarPanelLayout = new javax.swing.GroupLayout(addCarPanel);
        addCarPanel.setLayout(addCarPanelLayout);
        addCarPanelLayout.setHorizontalGroup(
            addCarPanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(addCarPanelLayout.createSequentialGroup()
                .addGroup(addCarPanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                    .addComponent(nameLabel)
                    .addComponent(nameTextField, javax.swing.GroupLayout.PREFERRED_SIZE, 265, javax.swing.GroupLayout.PREFERRED_SIZE)
                    .addComponent(modelText)
                    .addComponent(modelTextField, javax.swing.GroupLayout.PREFERRED_SIZE, 265, javax.swing.GroupLayout.PREFERRED_SIZE)
                    .addComponent(priceText)
                    .addComponent(priceTextField, javax.swing.GroupLayout.PREFERRED_SIZE, 265, javax.swing.GroupLayout.PREFERRED_SIZE))
                .addGap(0, 0, Short.MAX_VALUE))
            .addGroup(addCarPanelLayout.createSequentialGroup()
                .addGap(87, 87, 87)
                .addComponent(addBTN, javax.swing.GroupLayout.PREFERRED_SIZE, 100, javax.swing.GroupLayout.PREFERRED_SIZE)
                .addContainerGap(javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE))
        );
        addCarPanelLayout.setVerticalGroup(
            addCarPanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(addCarPanelLayout.createSequentialGroup()
                .addComponent(nameLabel)
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addComponent(nameTextField, javax.swing.GroupLayout.PREFERRED_SIZE, 30, javax.swing.GroupLayout.PREFERRED_SIZE)
                .addGap(47, 47, 47)
                .addComponent(modelText)
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addComponent(modelTextField, javax.swing.GroupLayout.PREFERRED_SIZE, 30, javax.swing.GroupLayout.PREFERRED_SIZE)
                .addGap(73, 73, 73)
                .addComponent(priceText)
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addComponent(priceTextField, javax.swing.GroupLayout.PREFERRED_SIZE, 30, javax.swing.GroupLayout.PREFERRED_SIZE)
                .addGap(59, 59, 59)
                .addComponent(addBTN, javax.swing.GroupLayout.PREFERRED_SIZE, 40, javax.swing.GroupLayout.PREFERRED_SIZE)
                .addGap(0, 102, Short.MAX_VALUE))
        );

        javax.swing.GroupLayout leftSideAdminLayout = new javax.swing.GroupLayout(leftSideAdmin);
        leftSideAdmin.setLayout(leftSideAdminLayout);
        leftSideAdminLayout.setHorizontalGroup(
            leftSideAdminLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addComponent(addCar, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
            .addComponent(addCarPanel, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
        );
        leftSideAdminLayout.setVerticalGroup(
            leftSideAdminLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(leftSideAdminLayout.createSequentialGroup()
                .addComponent(addCar, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                .addComponent(addCarPanel, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                .addGap(16, 16, 16))
        );

        middleSideAdmin.setBackground(new java.awt.Color(30, 41, 59));
        middleSideAdmin.setPreferredSize(new java.awt.Dimension(265, 500));

        deleteCAr.setFont(new java.awt.Font("Segoe UI", 1, 18)); // NOI18N
        deleteCAr.setForeground(new java.awt.Color(255, 255, 255));
        deleteCAr.setText("                   Delete Car");
        deleteCAr.setBorder(javax.swing.BorderFactory.createLineBorder(new java.awt.Color(0, 0, 0)));
        deleteCAr.setPreferredSize(new java.awt.Dimension(265, 40));
        deleteCAr.addMouseListener(new java.awt.event.MouseAdapter() {
            public void mouseClicked(java.awt.event.MouseEvent evt) {
                deleteCArMouseClicked(evt);
            }
        });

        carsBox.setBackground(new java.awt.Color(15, 23, 42));
        carsBox.setForeground(new java.awt.Color(255, 255, 255));
        carsBox.setModel(new javax.swing.DefaultComboBoxModel<>(new String[] { "Item 1", "Item 2", "Item 3", "Item 4" }));
        carsBox.addActionListener(this::carsBoxActionPerformed);

        deleteBTN.setBackground(new java.awt.Color(225, 29, 72));
        deleteBTN.setForeground(new java.awt.Color(255, 255, 255));
        deleteBTN.setText("Delete");

        javax.swing.GroupLayout deleteCarPanelLayout = new javax.swing.GroupLayout(deleteCarPanel);
        deleteCarPanel.setLayout(deleteCarPanelLayout);
        deleteCarPanelLayout.setHorizontalGroup(
            deleteCarPanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(deleteCarPanelLayout.createSequentialGroup()
                .addContainerGap()
                .addComponent(carsBox, 0, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                .addContainerGap())
            .addGroup(javax.swing.GroupLayout.Alignment.TRAILING, deleteCarPanelLayout.createSequentialGroup()
                .addContainerGap(javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                .addComponent(deleteBTN, javax.swing.GroupLayout.PREFERRED_SIZE, 100, javax.swing.GroupLayout.PREFERRED_SIZE)
                .addGap(82, 82, 82))
        );
        deleteCarPanelLayout.setVerticalGroup(
            deleteCarPanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(deleteCarPanelLayout.createSequentialGroup()
                .addGap(44, 44, 44)
                .addComponent(carsBox, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED, 276, Short.MAX_VALUE)
                .addComponent(deleteBTN, javax.swing.GroupLayout.PREFERRED_SIZE, 40, javax.swing.GroupLayout.PREFERRED_SIZE)
                .addGap(68, 68, 68))
        );

        javax.swing.GroupLayout middleSideAdminLayout = new javax.swing.GroupLayout(middleSideAdmin);
        middleSideAdmin.setLayout(middleSideAdminLayout);
        middleSideAdminLayout.setHorizontalGroup(
            middleSideAdminLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addComponent(deleteCAr, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
            .addComponent(deleteCarPanel, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
        );
        middleSideAdminLayout.setVerticalGroup(
            middleSideAdminLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(middleSideAdminLayout.createSequentialGroup()
                .addComponent(deleteCAr, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addComponent(deleteCarPanel, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE))
        );

        rightSideAdmin.setBackground(new java.awt.Color(30, 41, 59));
        rightSideAdmin.setPreferredSize(new java.awt.Dimension(265, 500));

        viewTransaction.setFont(new java.awt.Font("Segoe UI", 1, 18)); // NOI18N
        viewTransaction.setForeground(new java.awt.Color(255, 255, 255));
        viewTransaction.setText("             View Transaction");
        viewTransaction.setBorder(javax.swing.BorderFactory.createLineBorder(new java.awt.Color(0, 0, 0)));
        viewTransaction.addMouseListener(new java.awt.event.MouseAdapter() {
            public void mouseClicked(java.awt.event.MouseEvent evt) {
                viewTransactionMouseClicked(evt);
            }
        });

        javax.swing.GroupLayout viewTransactionLabelLayout = new javax.swing.GroupLayout(viewTransactionLabel);
        viewTransactionLabel.setLayout(viewTransactionLabelLayout);
        viewTransactionLabelLayout.setHorizontalGroup(
            viewTransactionLabelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGap(0, 0, Short.MAX_VALUE)
        );
        viewTransactionLabelLayout.setVerticalGroup(
            viewTransactionLabelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGap(0, 0, Short.MAX_VALUE)
        );

        scrollPanel.setBackground(new java.awt.Color(30, 41, 59));

        transactionNav.setBackground(new java.awt.Color(30, 41, 59));
        transactionNav.setPreferredSize(new java.awt.Dimension(255, 450));
        transactionNav.setLayout(new javax.swing.BoxLayout(transactionNav, javax.swing.BoxLayout.LINE_AXIS));

        item1.setBackground(new java.awt.Color(30, 41, 59));
        item1.setBorder(javax.swing.BorderFactory.createEmptyBorder(10, 10, 10, 10));
        item1.setForeground(new java.awt.Color(255, 255, 255));
        item1.setPreferredSize(new java.awt.Dimension(250, 60));

        javax.swing.GroupLayout item1Layout = new javax.swing.GroupLayout(item1);
        item1.setLayout(item1Layout);
        item1Layout.setHorizontalGroup(
            item1Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGap(0, 235, Short.MAX_VALUE)
        );
        item1Layout.setVerticalGroup(
            item1Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGap(0, 430, Short.MAX_VALUE)
        );

        transactionNav.add(item1);

        scrollPanel.setViewportView(transactionNav);

        javax.swing.GroupLayout rightSideAdminLayout = new javax.swing.GroupLayout(rightSideAdmin);
        rightSideAdmin.setLayout(rightSideAdminLayout);
        rightSideAdminLayout.setHorizontalGroup(
            rightSideAdminLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(javax.swing.GroupLayout.Alignment.TRAILING, rightSideAdminLayout.createSequentialGroup()
                .addGroup(rightSideAdminLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                    .addComponent(viewTransaction, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                    .addGroup(javax.swing.GroupLayout.Alignment.TRAILING, rightSideAdminLayout.createSequentialGroup()
                        .addGap(0, 0, Short.MAX_VALUE)
                        .addComponent(scrollPanel, javax.swing.GroupLayout.PREFERRED_SIZE, 261, javax.swing.GroupLayout.PREFERRED_SIZE)))
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addComponent(viewTransactionLabel, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE))
        );
        rightSideAdminLayout.setVerticalGroup(
            rightSideAdminLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(rightSideAdminLayout.createSequentialGroup()
                .addComponent(viewTransaction, javax.swing.GroupLayout.PREFERRED_SIZE, 40, javax.swing.GroupLayout.PREFERRED_SIZE)
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addGroup(rightSideAdminLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                    .addComponent(viewTransactionLabel, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                    .addComponent(scrollPanel, javax.swing.GroupLayout.DEFAULT_SIZE, 454, Short.MAX_VALUE)))
        );

        javax.swing.GroupLayout jPanel1Layout = new javax.swing.GroupLayout(jPanel1);
        jPanel1.setLayout(jPanel1Layout);
        jPanel1Layout.setHorizontalGroup(
            jPanel1Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(jPanel1Layout.createSequentialGroup()
                .addComponent(leftSideAdmin, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addComponent(middleSideAdmin, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addComponent(rightSideAdmin, javax.swing.GroupLayout.DEFAULT_SIZE, 267, Short.MAX_VALUE))
        );
        jPanel1Layout.setVerticalGroup(
            jPanel1Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(jPanel1Layout.createSequentialGroup()
                .addGroup(jPanel1Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                    .addComponent(leftSideAdmin, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                    .addComponent(middleSideAdmin, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                    .addComponent(rightSideAdmin, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE))
                .addGap(0, 0, Short.MAX_VALUE))
        );

        javax.swing.GroupLayout layout = new javax.swing.GroupLayout(getContentPane());
        getContentPane().setLayout(layout);
        layout.setHorizontalGroup(
            layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addComponent(jPanel1, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
        );
        layout.setVerticalGroup(
            layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(layout.createSequentialGroup()
                .addComponent(jPanel1, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                .addGap(0, 0, Short.MAX_VALUE))
        );

        pack();
    }// </editor-fold>//GEN-END:initComponents

    private void addCarMouseClicked(java.awt.event.MouseEvent evt) {//GEN-FIRST:event_addCarMouseClicked
        addCarPanel.setVisible(true);
    }//GEN-LAST:event_addCarMouseClicked

    private void deleteCArMouseClicked(java.awt.event.MouseEvent evt) {//GEN-FIRST:event_deleteCArMouseClicked
       deleteCarPanel.setVisible(true);
    }//GEN-LAST:event_deleteCArMouseClicked

    private void carsBoxActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_carsBoxActionPerformed
        
    }//GEN-LAST:event_carsBoxActionPerformed

    private void viewTransactionMouseClicked(java.awt.event.MouseEvent evt) {//GEN-FIRST:event_viewTransactionMouseClicked
        loadTransactions();
    }//GEN-LAST:event_viewTransactionMouseClicked

    private void addBTNActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_addBTNActionPerformed
        // TODO add your handling code here:
    }//GEN-LAST:event_addBTNActionPerformed

    public static void main(String args[]) {
        java.awt.EventQueue.invokeLater(() -> new Admins().setVisible(true));
    }

    public javax.swing.JComboBox<String> getCarsBox() {
        return carsBox;
    }

    public javax.swing.JPanel getTransactionNav() {
        return transactionNav;
    }
    
    // Variables declaration - do not modify//GEN-BEGIN:variables
    private javax.swing.JButton addBTN;
    private javax.swing.JLabel addCar;
    private javax.swing.JPanel addCarPanel;
    private javax.swing.JComboBox<String> carsBox;
    private javax.swing.JButton deleteBTN;
    private javax.swing.JLabel deleteCAr;
    private javax.swing.JPanel deleteCarPanel;
    private javax.swing.JPanel item1;
    private javax.swing.JPanel jPanel1;
    private javax.swing.JPanel leftSideAdmin;
    private javax.swing.JPanel middleSideAdmin;
    private javax.swing.JLabel modelText;
    private javax.swing.JTextField modelTextField;
    private javax.swing.JLabel nameLabel;
    private javax.swing.JTextField nameTextField;
    private javax.swing.JLabel priceText;
    private javax.swing.JTextField priceTextField;
    private javax.swing.JPanel rightSideAdmin;
    private javax.swing.JScrollPane scrollPanel;
    private javax.swing.JPanel transactionNav;
    private javax.swing.JLabel viewTransaction;
    private javax.swing.JPanel viewTransactionLabel;
    // End of variables declaration//GEN-END:variables
}
