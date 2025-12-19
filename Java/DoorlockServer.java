import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.net.ServerSocket;
import java.net.Socket;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class DoorlockServer {

    private final int port;
    private final List<PrintWriter> clients = Collections.synchronizedList(new ArrayList<>());
    private DoorlockListener listener;

    public DoorlockServer(int port) {
        this.port = port;
    }

    public interface DoorlockListener {
        void onDoorlockEvent(String event);
    }

    public void addDoorlockListener(DoorlockListener l) {
        this.listener = l;
    }

    public void start() {
        Thread t = new Thread(() -> {
            try (ServerSocket serverSocket = new ServerSocket(port)) {
                System.out.println("문 이벤트 서버 시작, 포트: " + port);

                while (true) {
                    Socket client = serverSocket.accept();
                    System.out.println("문 이벤트 클라이언트 연결 성공: " + client.getInetAddress());
                    PrintWriter writer = new PrintWriter(client.getOutputStream(), true);
                    clients.add(writer);

                    Thread reader = new Thread(() -> handleClient(client, writer), "door-event-reader");
                    reader.setDaemon(true);
                    reader.start();
                }

            } catch (Exception e) {
                e.printStackTrace();
            }
        }, "문 이벤트 서버");
        t.setDaemon(true);
        t.start();
    }

    private void handleClient(Socket client, PrintWriter writer) {
        try (BufferedReader in = new BufferedReader(new InputStreamReader(client.getInputStream()))) {
            String line;
            while ((line = in.readLine()) != null) {
                String evt = line.trim();
                if (evt.isEmpty()) continue;
                System.out.println("문 이벤트: " + evt);
                if (listener != null) listener.onDoorlockEvent(evt);
                broadcast(evt, writer);
            }
        } catch (Exception ignored) {
        } finally {
            clients.remove(writer);
            try { client.close(); } catch (Exception ignored) {}
            System.out.println("문 이벤트 클라이언트 연결 종료");
        }
    }

    private void broadcast(String evt, PrintWriter sender) {
        synchronized (clients) {
            for (PrintWriter out : new ArrayList<>(clients)) {
                try {
                    if (out != sender) out.println(evt);
                } catch (Exception e) {
                    clients.remove(out);
                }
            }
        }
    }

    public void broadcast(String evt) {
        broadcast(evt, null);
    }
}
