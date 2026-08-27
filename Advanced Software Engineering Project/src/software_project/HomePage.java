package software_project;

import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import javax.swing.ImageIcon;
import javax.swing.JOptionPane;

public class HomePage extends javax.swing.JFrame {
    String selectedCar = "";
    int selectedCarId = -1;
    int currentUserId;
    
    private static final java.util.logging.Logger logger = java.util.logging.Logger.getLogger(HomePage.class.getName());

    public HomePage(int userId) {
        initComponents();
        photo1.setIcon(new ImageIcon(getClass().getResource("/software_project/icons/dodge.jpg")));
        photo2.setIcon(new ImageIcon(getClass().getResource("/software_project/icons/bugatti.jpg")));
        photo3.setIcon(new ImageIcon(getClass().getResource("/software_project/icons/ferrari.jpg")));
        photo4.setIcon(new ImageIcon(getClass().getResource("/software_project/icons/lamborghini.jpg")));
        photo5.setIcon(new ImageIcon(getClass().getResource("/software_project/icons/mercedes.jpg")));
        photo6.setIcon(new ImageIcon(getClass().getResource("/software_project/icons/rolls.jpg")));
        showCar(-1, "Choose a car", "-", "-");
        rentBTN.setEnabled(false);
        buyBTN.setEnabled(false);
        this.currentUserId = userId;
    }

    @SuppressWarnings("unchecked")
    // <editor-fold defaultstate="collapsed" desc="Generated Code">//GEN-BEGIN:initComponents
    private void initComponents() {

        jPanel1 = new javax.swing.JPanel();
        headerPanel = new javax.swing.JPanel();
        header = new javax.swing.JLabel();
        rightSide = new javax.swing.JPanel();
        nameCar = new javax.swing.JLabel();
        modelCar = new javax.swing.JLabel();
        priceCar = new javax.swing.JLabel();
        details = new javax.swing.JLabel();
        rentBTN = new javax.swing.JButton();
        buyBTN = new javax.swing.JButton();
        name = new javax.swing.JLabel();
        jLabel1 = new javax.swing.JLabel();
        jLabel2 = new javax.swing.JLabel();
        leftSide = new javax.swing.JPanel();
        photo1 = new javax.swing.JLabel();
        photo2 = new javax.swing.JLabel();
        photo3 = new javax.swing.JLabel();
        photo4 = new javax.swing.JLabel();
        photo5 = new javax.swing.JLabel();
        photo6 = new javax.swing.JLabel();

        setDefaultCloseOperation(javax.swing.WindowConstants.EXIT_ON_CLOSE);
        setTitle("Home");

        jPanel1.setPreferredSize(new java.awt.Dimension(800, 500));

        headerPanel.setBackground(new java.awt.Color(225, 29, 72));
        headerPanel.setPreferredSize(new java.awt.Dimension(800, 50));

        header.setBackground(new java.awt.Color(167, 243, 208));
        header.setFont(new java.awt.Font("Segoe Script", 0, 36)); // NOI18N
        header.setForeground(new java.awt.Color(176, 228, 204));
        header.setText("Car Rental System");
        header.setPreferredSize(new java.awt.Dimension(38, 50));

        javax.swing.GroupLayout headerPanelLayout = new javax.swing.GroupLayout(headerPanel);
        headerPanel.setLayout(headerPanelLayout);
        headerPanelLayout.setHorizontalGroup(
            headerPanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(headerPanelLayout.createSequentialGroup()
                .addGap(222, 222, 222)
                .addComponent(header, javax.swing.GroupLayout.PREFERRED_SIZE, 368, javax.swing.GroupLayout.PREFERRED_SIZE)
                .addContainerGap(184, Short.MAX_VALUE))
        );
        headerPanelLayout.setVerticalGroup(
            headerPanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(headerPanelLayout.createSequentialGroup()
                .addComponent(header, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                .addContainerGap())
        );

        rightSide.setBackground(new java.awt.Color(30, 41, 59));
        rightSide.setPreferredSize(new java.awt.Dimension(300, 450));

        nameCar.setBackground(new java.awt.Color(15, 23, 42));
        nameCar.setForeground(new java.awt.Color(255, 255, 255));
        nameCar.setBorder(javax.swing.BorderFactory.createLineBorder(new java.awt.Color(0, 0, 0)));
        nameCar.setOpaque(true);

        modelCar.setBackground(new java.awt.Color(15, 23, 42));
        modelCar.setForeground(new java.awt.Color(255, 255, 255));
        modelCar.setBorder(javax.swing.BorderFactory.createLineBorder(new java.awt.Color(0, 0, 0)));
        modelCar.setOpaque(true);

        priceCar.setBackground(new java.awt.Color(15, 23, 42));
        priceCar.setForeground(new java.awt.Color(255, 255, 255));
        priceCar.setBorder(javax.swing.BorderFactory.createLineBorder(new java.awt.Color(0, 0, 0)));
        priceCar.setOpaque(true);

        details.setFont(new java.awt.Font("Segoe Script", 1, 18)); // NOI18N
        details.setText("Details");

        rentBTN.setBackground(new java.awt.Color(225, 29, 72));
        rentBTN.setForeground(new java.awt.Color(255, 255, 255));
        rentBTN.setText("Rent");
        rentBTN.addActionListener(this::rentBTNActionPerformed);

        buyBTN.setBackground(new java.awt.Color(225, 29, 72));
        buyBTN.setForeground(new java.awt.Color(255, 255, 255));
        buyBTN.setText("Buy");
        buyBTN.addActionListener(this::buyBTNActionPerformed);

        name.setBackground(new java.awt.Color(176, 228, 204));
        name.setFont(new java.awt.Font("Segoe UI", 1, 14)); // NOI18N
        name.setForeground(new java.awt.Color(203, 213, 225));
        name.setText("Name:");

        jLabel1.setFont(new java.awt.Font("Segoe UI", 1, 14)); // NOI18N
        jLabel1.setForeground(new java.awt.Color(203, 213, 225));
        jLabel1.setText("Model:");

        jLabel2.setFont(new java.awt.Font("Segoe UI", 1, 18)); // NOI18N
        jLabel2.setForeground(new java.awt.Color(203, 213, 225));
        jLabel2.setText("Price:");

        javax.swing.GroupLayout rightSideLayout = new javax.swing.GroupLayout(rightSide);
        rightSide.setLayout(rightSideLayout);
        rightSideLayout.setHorizontalGroup(
            rightSideLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(rightSideLayout.createSequentialGroup()
                .addContainerGap()
                .addGroup(rightSideLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                    .addGroup(rightSideLayout.createSequentialGroup()
                        .addComponent(name, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                        .addGap(218, 218, 218))
                    .addGroup(rightSideLayout.createSequentialGroup()
                        .addGap(0, 0, Short.MAX_VALUE)
                        .addComponent(details)
                        .addGap(125, 125, 125))
                    .addGroup(rightSideLayout.createSequentialGroup()
                        .addGroup(rightSideLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                            .addComponent(priceCar, javax.swing.GroupLayout.PREFERRED_SIZE, 200, javax.swing.GroupLayout.PREFERRED_SIZE)
                            .addComponent(jLabel2)
                            .addComponent(jLabel1)
                            .addGroup(rightSideLayout.createSequentialGroup()
                                .addComponent(rentBTN, javax.swing.GroupLayout.PREFERRED_SIZE, 112, javax.swing.GroupLayout.PREFERRED_SIZE)
                                .addGap(18, 18, 18)
                                .addComponent(buyBTN, javax.swing.GroupLayout.PREFERRED_SIZE, 112, javax.swing.GroupLayout.PREFERRED_SIZE))
                            .addComponent(modelCar, javax.swing.GroupLayout.PREFERRED_SIZE, 200, javax.swing.GroupLayout.PREFERRED_SIZE)
                            .addComponent(nameCar, javax.swing.GroupLayout.PREFERRED_SIZE, 200, javax.swing.GroupLayout.PREFERRED_SIZE))
                        .addGap(0, 0, Short.MAX_VALUE))))
        );
        rightSideLayout.setVerticalGroup(
            rightSideLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(rightSideLayout.createSequentialGroup()
                .addComponent(details, javax.swing.GroupLayout.PREFERRED_SIZE, 50, javax.swing.GroupLayout.PREFERRED_SIZE)
                .addGap(18, 18, 18)
                .addComponent(name, javax.swing.GroupLayout.PREFERRED_SIZE, 30, javax.swing.GroupLayout.PREFERRED_SIZE)
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addComponent(nameCar, javax.swing.GroupLayout.PREFERRED_SIZE, 30, javax.swing.GroupLayout.PREFERRED_SIZE)
                .addGap(18, 18, 18)
                .addComponent(jLabel1, javax.swing.GroupLayout.PREFERRED_SIZE, 30, javax.swing.GroupLayout.PREFERRED_SIZE)
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addComponent(modelCar, javax.swing.GroupLayout.PREFERRED_SIZE, 30, javax.swing.GroupLayout.PREFERRED_SIZE)
                .addGap(18, 18, 18)
                .addComponent(jLabel2, javax.swing.GroupLayout.PREFERRED_SIZE, 30, javax.swing.GroupLayout.PREFERRED_SIZE)
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addComponent(priceCar, javax.swing.GroupLayout.PREFERRED_SIZE, 30, javax.swing.GroupLayout.PREFERRED_SIZE)
                .addGap(18, 18, 18)
                .addGroup(rightSideLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
                    .addComponent(rentBTN, javax.swing.GroupLayout.PREFERRED_SIZE, 40, javax.swing.GroupLayout.PREFERRED_SIZE)
                    .addComponent(buyBTN, javax.swing.GroupLayout.PREFERRED_SIZE, 40, javax.swing.GroupLayout.PREFERRED_SIZE))
                .addContainerGap(90, Short.MAX_VALUE))
        );

        leftSide.setBackground(new java.awt.Color(15, 23, 42));
        leftSide.setPreferredSize(new java.awt.Dimension(500, 450));

        photo1.setIcon(new javax.swing.ImageIcon(getClass().getResource("/software_project/icons/dodge.jpg"))); // NOI18N
        photo1.addMouseListener(new java.awt.event.MouseAdapter() {
            public void mouseClicked(java.awt.event.MouseEvent evt) {
                photo1MouseClicked(evt);
            }
        });

        photo2.setIcon(new javax.swing.ImageIcon(getClass().getResource("/software_project/icons/bugatti.jpg"))); // NOI18N
        photo2.addMouseListener(new java.awt.event.MouseAdapter() {
            public void mouseClicked(java.awt.event.MouseEvent evt) {
                photo2MouseClicked(evt);
            }
        });

        photo3.setIcon(new javax.swing.ImageIcon(getClass().getResource("/software_project/icons/ferrari.jpg"))); // NOI18N
        photo3.addMouseListener(new java.awt.event.MouseAdapter() {
            public void mouseClicked(java.awt.event.MouseEvent evt) {
                photo3MouseClicked(evt);
            }
        });

        photo4.setIcon(new javax.swing.ImageIcon(getClass().getResource("/software_project/icons/lamborghini.jpg"))); // NOI18N
        photo4.addMouseListener(new java.awt.event.MouseAdapter() {
            public void mouseClicked(java.awt.event.MouseEvent evt) {
                photo4MouseClicked(evt);
            }
        });

        photo5.setIcon(new javax.swing.ImageIcon(getClass().getResource("/software_project/icons/mercedes.jpg"))); // NOI18N
        photo5.addMouseListener(new java.awt.event.MouseAdapter() {
            public void mouseClicked(java.awt.event.MouseEvent evt) {
                photo5MouseClicked(evt);
            }
        });

        photo6.setIcon(new javax.swing.ImageIcon(getClass().getResource("/software_project/icons/rolls.jpg"))); // NOI18N
        photo6.setText("jLabel1");
        photo6.addMouseListener(new java.awt.event.MouseAdapter() {
            public void mouseClicked(java.awt.event.MouseEvent evt) {
                photo6MouseClicked(evt);
            }
        });

        javax.swing.GroupLayout leftSideLayout = new javax.swing.GroupLayout(leftSide);
        leftSide.setLayout(leftSideLayout);
        leftSideLayout.setHorizontalGroup(
            leftSideLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(leftSideLayout.createSequentialGroup()
                .addGroup(leftSideLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                    .addComponent(photo1, javax.swing.GroupLayout.PREFERRED_SIZE, 200, javax.swing.GroupLayout.PREFERRED_SIZE)
                    .addComponent(photo3, javax.swing.GroupLayout.PREFERRED_SIZE, 200, javax.swing.GroupLayout.PREFERRED_SIZE)
                    .addComponent(photo5, javax.swing.GroupLayout.PREFERRED_SIZE, 200, javax.swing.GroupLayout.PREFERRED_SIZE))
                .addGap(106, 106, 106)
                .addGroup(leftSideLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                    .addComponent(photo6, javax.swing.GroupLayout.PREFERRED_SIZE, 200, javax.swing.GroupLayout.PREFERRED_SIZE)
                    .addGroup(leftSideLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING, false)
                        .addComponent(photo2, javax.swing.GroupLayout.PREFERRED_SIZE, 200, javax.swing.GroupLayout.PREFERRED_SIZE)
                        .addComponent(photo4, javax.swing.GroupLayout.PREFERRED_SIZE, 0, Short.MAX_VALUE)))
                .addGap(0, 0, Short.MAX_VALUE))
        );
        leftSideLayout.setVerticalGroup(
            leftSideLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(leftSideLayout.createSequentialGroup()
                .addGroup(leftSideLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                    .addComponent(photo2, javax.swing.GroupLayout.PREFERRED_SIZE, 130, javax.swing.GroupLayout.PREFERRED_SIZE)
                    .addComponent(photo1, javax.swing.GroupLayout.PREFERRED_SIZE, 130, javax.swing.GroupLayout.PREFERRED_SIZE))
                .addGap(18, 18, 18)
                .addGroup(leftSideLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                    .addComponent(photo3, javax.swing.GroupLayout.PREFERRED_SIZE, 130, javax.swing.GroupLayout.PREFERRED_SIZE)
                    .addComponent(photo4, javax.swing.GroupLayout.PREFERRED_SIZE, 130, javax.swing.GroupLayout.PREFERRED_SIZE))
                .addGap(18, 18, 18)
                .addGroup(leftSideLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                    .addComponent(photo5, javax.swing.GroupLayout.PREFERRED_SIZE, 130, javax.swing.GroupLayout.PREFERRED_SIZE)
                    .addComponent(photo6, javax.swing.GroupLayout.PREFERRED_SIZE, 130, javax.swing.GroupLayout.PREFERRED_SIZE))
                .addGap(0, 24, Short.MAX_VALUE))
        );

        javax.swing.GroupLayout jPanel1Layout = new javax.swing.GroupLayout(jPanel1);
        jPanel1.setLayout(jPanel1Layout);
        jPanel1Layout.setHorizontalGroup(
            jPanel1Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addComponent(headerPanel, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
            .addGroup(jPanel1Layout.createSequentialGroup()
                .addComponent(leftSide, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addComponent(rightSide, javax.swing.GroupLayout.DEFAULT_SIZE, 294, Short.MAX_VALUE))
        );
        jPanel1Layout.setVerticalGroup(
            jPanel1Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(jPanel1Layout.createSequentialGroup()
                .addComponent(headerPanel, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addGroup(jPanel1Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                    .addComponent(rightSide, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                    .addComponent(leftSide, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)))
        );

        javax.swing.GroupLayout layout = new javax.swing.GroupLayout(getContentPane());
        getContentPane().setLayout(layout);
        layout.setHorizontalGroup(
            layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addComponent(jPanel1, javax.swing.GroupLayout.Alignment.TRAILING, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
        );
        layout.setVerticalGroup(
            layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(javax.swing.GroupLayout.Alignment.TRAILING, layout.createSequentialGroup()
                .addComponent(jPanel1, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                .addGap(0, 0, Short.MAX_VALUE))
        );

        pack();
    }// </editor-fold>//GEN-END:initComponents

    void showCar(int id, String name, String model, String price) {
        nameCar.setText(name);
        modelCar.setText(model);
        priceCar.setText(price);
        selectedCar = name;
        selectedCarId = id;

        if(id != -1) {
            rentBTN.setEnabled(true);
            buyBTN.setEnabled(true);
        }
    }
    
    private void processTransaction(String type, String newStatus) {
        if(selectedCarId == -1) {
            JOptionPane.showMessageDialog(this, "Please select a car first!");
            return;
        }

        Connection con = null;

        try {
            con = DBConnection.connect();

            String check = "SELECT status FROM Cars WHERE id = ?";
            PreparedStatement pstCheck = con.prepareStatement(check);
            pstCheck.setInt(1, selectedCarId);

            ResultSet rs = pstCheck.executeQuery();

            if(rs.next()) {
                String status = rs.getString("status");

                if(!status.equals("available")) {
                    JOptionPane.showMessageDialog(this, "Car is not available!");
                    return;
                }
            }

            String query = "INSERT INTO Transactions (user_id, car_id, type) VALUES (?, ?, ?)";
            PreparedStatement pst = con.prepareStatement(query);
            pst.setInt(1, currentUserId);
            pst.setInt(2, selectedCarId);
            pst.setString(3, type);
            pst.executeUpdate();

            String update = "UPDATE Cars SET status = ? WHERE id = ?";
            PreparedStatement pst2 = con.prepareStatement(update);
            pst2.setString(1, newStatus);
            pst2.setInt(2, selectedCarId);
            pst2.executeUpdate();

            if(type.equals("rent")) {
                JOptionPane.showMessageDialog(this, "Car rented successfully!");
            } else {
                JOptionPane.showMessageDialog(this, "Car purchased successfully!");
            }

        } catch (Exception e) {
            JOptionPane.showMessageDialog(this, "Error: " + e.getMessage());
        } finally {
            try {
                if(con != null) con.close();
            } catch(Exception e){}
        }
    }
    
    private void photo1MouseClicked(java.awt.event.MouseEvent evt) {//GEN-FIRST:event_photo1MouseClicked
        showCar(1, "Dodge Charger", "2022", "$150 / day");
    }//GEN-LAST:event_photo1MouseClicked

    private void photo2MouseClicked(java.awt.event.MouseEvent evt) {//GEN-FIRST:event_photo2MouseClicked
        showCar(2, "Bugatti Chiron", "2023", "$1000 / day");
    }//GEN-LAST:event_photo2MouseClicked

    private void photo3MouseClicked(java.awt.event.MouseEvent evt) {//GEN-FIRST:event_photo3MouseClicked
        showCar(3, "Ferrari SF90", "2022", "$800 / day");
    }//GEN-LAST:event_photo3MouseClicked

    private void photo4MouseClicked(java.awt.event.MouseEvent evt) {//GEN-FIRST:event_photo4MouseClicked
        showCar(4, "Lamborghini Huracan", "2021", "$700 / day");
    }//GEN-LAST:event_photo4MouseClicked

    private void photo5MouseClicked(java.awt.event.MouseEvent evt) {//GEN-FIRST:event_photo5MouseClicked
        showCar(5, "Mercedes AMG", "2022", "$500 / day");
    }//GEN-LAST:event_photo5MouseClicked

    private void photo6MouseClicked(java.awt.event.MouseEvent evt) {//GEN-FIRST:event_photo6MouseClicked
        showCar(6, "Rolls Royce Phantom", "2023", "$1200 / day");
    }//GEN-LAST:event_photo6MouseClicked

    private void rentBTNActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_rentBTNActionPerformed
         processTransaction("rent", "rented");
    }//GEN-LAST:event_rentBTNActionPerformed

    private void buyBTNActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_buyBTNActionPerformed
        processTransaction("buy", "sold");
    }//GEN-LAST:event_buyBTNActionPerformed

    public static void main(String args[]) {
        //<editor-fold defaultstate="collapsed" desc=" Look and feel setting code (optional) ">
        /* If Nimbus (introduced in Java SE 6) is not available, stay with the default look and feel.
         * For details see http://download.oracle.com/javase/tutorial/uiswing/lookandfeel/plaf.html 
         */
        try {
            for (javax.swing.UIManager.LookAndFeelInfo info : javax.swing.UIManager.getInstalledLookAndFeels()) {
                if ("Nimbus".equals(info.getName())) {
                    javax.swing.UIManager.setLookAndFeel(info.getClassName());
                    break;
                }
            }
        } catch (ReflectiveOperationException | javax.swing.UnsupportedLookAndFeelException ex) {
            logger.log(java.util.logging.Level.SEVERE, null, ex);
        }
        //</editor-fold>

        java.awt.EventQueue.invokeLater(() -> new HomePage(1).setVisible(true));
    }
    
    public javax.swing.JLabel getNameCar() {
        return nameCar;
    }

    public javax.swing.JLabel getModelCar() {
        return modelCar;
    }

    public javax.swing.JLabel getPriceCar() {
        return priceCar;
    }

    public javax.swing.JButton getRentBTN() {
        return rentBTN;
    }

    public javax.swing.JButton getBuyBTN() {
        return buyBTN;
    }

    // Variables declaration - do not modify//GEN-BEGIN:variables
    private javax.swing.JButton buyBTN;
    private javax.swing.JLabel details;
    private javax.swing.JLabel header;
    private javax.swing.JPanel headerPanel;
    private javax.swing.JLabel jLabel1;
    private javax.swing.JLabel jLabel2;
    private javax.swing.JPanel jPanel1;
    private javax.swing.JPanel leftSide;
    private javax.swing.JLabel modelCar;
    private javax.swing.JLabel name;
    private javax.swing.JLabel nameCar;
    private javax.swing.JLabel photo1;
    private javax.swing.JLabel photo2;
    private javax.swing.JLabel photo3;
    private javax.swing.JLabel photo4;
    private javax.swing.JLabel photo5;
    private javax.swing.JLabel photo6;
    private javax.swing.JLabel priceCar;
    private javax.swing.JButton rentBTN;
    private javax.swing.JPanel rightSide;
    // End of variables declaration//GEN-END:variables
}
