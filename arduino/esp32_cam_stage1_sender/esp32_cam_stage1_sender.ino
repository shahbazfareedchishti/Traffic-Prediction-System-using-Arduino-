#include "esp_camera.h"
#include <WiFi.h>
#include <HTTPClient.h>

// -------- Camera model (AI Thinker ESP32-CAM) --------
#define PWDN_GPIO_NUM     32
#define RESET_GPIO_NUM    -1
#define XCLK_GPIO_NUM      0
#define SIOD_GPIO_NUM     26
#define SIOC_GPIO_NUM     27

#define Y9_GPIO_NUM       35
#define Y8_GPIO_NUM       34
#define Y7_GPIO_NUM       39
#define Y6_GPIO_NUM       36
#define Y5_GPIO_NUM       21
#define Y4_GPIO_NUM       19
#define Y3_GPIO_NUM       18
#define Y2_GPIO_NUM        5
#define VSYNC_GPIO_NUM    25
#define HREF_GPIO_NUM     23
#define PCLK_GPIO_NUM     22

// -------- User configuration --------
const char* WIFI_SSID = "YOUR_WIFI_NAME";
const char* WIFI_PASSWORD = "YOUR_WIFI_PASSWORD";

// Use your laptop/server local IP running stage1_receiver.py
const char* STAGE1_BASE_URL = "http://192.168.1.100:7000";

const char* DEVICE_ID = "esp32cam_01";
const char* LOCATION_NAME = "main_gate";

const unsigned long FRAME_INTERVAL_MS = 1000;     // 1 frame per second
const unsigned long RECORD_DURATION_MS = 120000;  // 2 minutes per session
const unsigned long PAUSE_BETWEEN_SESSIONS_MS = 10000;

String sessionId = "";
unsigned long sessionStartMs = 0;
unsigned long lastFrameMs = 0;
bool sessionActive = false;


bool initCamera() {
  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;
  config.pin_sccb_sda = SIOD_GPIO_NUM;
  config.pin_sccb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_JPEG;

  if (psramFound()) {
    config.frame_size = FRAMESIZE_VGA;
    config.jpeg_quality = 12;
    config.fb_count = 2;
  } else {
    config.frame_size = FRAMESIZE_QVGA;
    config.jpeg_quality = 15;
    config.fb_count = 1;
  }

  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("Camera init failed (0x%x)\n", err);
    return false;
  }
  return true;
}


void connectWiFi() {
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  Serial.print("Connecting to WiFi");
  while (WiFi.status() != WL_CONNECTED) {
    delay(400);
    Serial.print(".");
  }
  Serial.println();
  Serial.print("WiFi connected, IP: ");
  Serial.println(WiFi.localIP());
}


String parseSessionId(String response) {
  // Expected: "session_id=<id>;frame_index=<n>"
  int start = response.indexOf("session_id=");
  if (start < 0) {
    return "";
  }
  start += 11;  // strlen("session_id=")
  int end = response.indexOf(";", start);
  if (end < 0) {
    end = response.length();
  }
  return response.substring(start, end);
}


bool uploadFrame() {
  if (WiFi.status() != WL_CONNECTED) {
    connectWiFi();
  }

  camera_fb_t* fb = esp_camera_fb_get();
  if (!fb) {
    Serial.println("Failed to capture frame");
    return false;
  }

  WiFiClient client;
  HTTPClient http;
  String frameUrl = String(STAGE1_BASE_URL) + "/stage1/frame";
  bool ok = false;

  if (http.begin(client, frameUrl)) {
    http.addHeader("Content-Type", "image/jpeg");
    http.addHeader("X-Device-Id", DEVICE_ID);
    http.addHeader("X-Location", LOCATION_NAME);
    http.addHeader("X-Timestamp-Ms", String(millis()));
    if (sessionId.length() > 0) {
      http.addHeader("X-Session-Id", sessionId);
    }

    int code = http.POST(fb->buf, fb->len);
    String response = http.getString();
    http.end();

    if (code > 0 && code < 300) {
      String receivedSessionId = parseSessionId(response);
      if (receivedSessionId.length() > 0) {
        sessionId = receivedSessionId;
      }
      Serial.printf("Frame uploaded: code=%d, session=%s, size=%u\n", code, sessionId.c_str(), fb->len);
      ok = true;
    } else {
      Serial.printf("Frame upload failed: code=%d response=%s\n", code, response.c_str());
    }
  } else {
    Serial.println("HTTP begin failed for /stage1/frame");
  }

  esp_camera_fb_return(fb);
  return ok;
}


void closeSession() {
  if (sessionId.length() == 0) {
    return;
  }

  WiFiClient client;
  HTTPClient http;
  String closeUrl = String(STAGE1_BASE_URL) + "/stage1/close";
  if (!http.begin(client, closeUrl)) {
    Serial.println("HTTP begin failed for /stage1/close");
    return;
  }

  String payload = "session_id=" + sessionId;
  http.addHeader("Content-Type", "application/x-www-form-urlencoded");
  int code = http.POST(payload);
  String response = http.getString();
  http.end();

  Serial.printf("Session close: code=%d, session=%s\n", code, sessionId.c_str());
  Serial.println(response);
}


void startNewSession() {
  sessionId = "";
  sessionStartMs = millis();
  lastFrameMs = 0;
  sessionActive = true;
  Serial.println("Started new recording session");
}


void setup() {
  Serial.begin(115200);
  delay(1000);
  Serial.println();
  Serial.println("ESP32-CAM Stage-1 Sender starting...");

  if (!initCamera()) {
    while (true) {
      delay(1000);
    }
  }
  connectWiFi();
  startNewSession();
}


void loop() {
  if (!sessionActive) {
    delay(250);
    return;
  }

  unsigned long nowMs = millis();
  if (nowMs - sessionStartMs >= RECORD_DURATION_MS) {
    closeSession();
    sessionActive = false;
    Serial.println("Session finished, waiting before next session...");
    delay(PAUSE_BETWEEN_SESSIONS_MS);
    startNewSession();
    return;
  }

  if (lastFrameMs == 0 || nowMs - lastFrameMs >= FRAME_INTERVAL_MS) {
    uploadFrame();
    lastFrameMs = nowMs;
  }

  delay(20);
}
