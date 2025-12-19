import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.net.ServerSocket;
import java.net.Socket;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * 브로드 캐스트로 파이썬으로 전송
 * 커맨드 리스너로 GUI에 알려주기
 */
public class TcpServer {

    private final int port;
    private final List<PrintWriter> clients = Collections.synchronizedList(new ArrayList<>());
    private CommandListener commandListener;

    public TcpServer(int port) {
        this.port = port;
    }

    public interface CommandListener {
        void onCommand(String cmd);
    }

    public void addCommandListener(CommandListener listener) {
        this.commandListener = listener;
    }

    public void start() {
        Thread t = new Thread(() -> {
            try (ServerSocket serverSocket = new ServerSocket(port)) {
                System.out.println("[JAVA] Command server listening on port " + port);

                while (true) {
                    Socket clientSocket = serverSocket.accept();
                    System.out.println("[JAVA] Command client connected: " + clientSocket.getInetAddress());

                    PrintWriter writer = new PrintWriter(clientSocket.getOutputStream(), true);
                    clients.add(writer);

                    Thread reader = new Thread(() -> handleClient(clientSocket, writer), "command-reader");
                    reader.setDaemon(true);
                    reader.start();
                }
            } catch (IOException e) {
                e.printStackTrace();
            }
        }, "command-server");
        t.setDaemon(true);
        t.start();
    }

    private void handleClient(Socket clientSocket, PrintWriter writer) {
        try (BufferedReader in = new BufferedReader(new InputStreamReader(clientSocket.getInputStream()))) {
            String line;
            while ((line = in.readLine()) != null) {
                String cmd = line.trim();
                if (cmd.isEmpty()) continue;
                System.out.println("[JAVA] Command received: " + cmd);
                if (commandListener != null) {
                    commandListener.onCommand(cmd);
                }
                broadcast(cmd, writer);
            }
        } catch (IOException ignored) {
        } finally {
            clients.remove(writer);
            System.out.println("[JAVA] Command client disconnected");
            try { clientSocket.close(); } catch (Exception ignored) {}
        }
    }

    // 모든 클라이언트로 발송
    private void broadcast(String cmd, PrintWriter sender) {
        synchronized (clients) {
            for (PrintWriter out : new ArrayList<>(clients)) {
                try {
                    if (out != sender) out.println(cmd);
                } catch (Exception e) {
                    clients.remove(out);
                }
            }
        }
    }

    // GUI에서 받는 메시지를 모든 클라이언트로 전송
    public void sendCommand(String cmd) {
        broadcast(cmd, null);
    }
}
