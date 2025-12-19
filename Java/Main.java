import javax.swing.SwingUtilities;

public class Main {

    public static void main(String[] args) {

        // 포트 설정 (파이썬 코드와 일치해야 함)
        int COMMAND_PORT = 39186;
        int SENSOR_PORT  = 39187;
        int DOOR_PORT    = 39189;

        System.out.println("=== [JAVA] Smart Home System Starting ===");

        // 1. 서버 객체 생성
        TcpServer commandServer = new TcpServer(COMMAND_PORT);
        SensorTcpServer sensorServer = new SensorTcpServer(SENSOR_PORT);
        DoorlockServer doorlockServer = new DoorlockServer(DOOR_PORT);

        // 2. 서버 실행 ( 중요 수정: 각각 별도 스레드에서 실행)
        // 이렇게 해야 하나가 연결 대기 중이라도 다른 코드(GUI)가 멈추지 않습니다.
        new Thread(() -> commandServer.start()).start();
        new Thread(() -> sensorServer.start()).start();
        new Thread(() -> doorlockServer.start()).start();

        // 3. GUI 실행 (Swing 스레드 안전성 확보)
        SwingUtilities.invokeLater(() -> {
            SmartHomeGUI gui = new SmartHomeGUI(commandServer, sensorServer, doorlockServer);
            gui.showWindow();
            System.out.println("[JAVA] GUI 창이 표시되었습니다.");
        });
    }
}