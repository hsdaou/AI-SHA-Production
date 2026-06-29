// Soil Moisture Sensor - Arduino Sketch
// MH-sensor-series module connected to:
//   AO -> A0 (analog raw value)
//   DO -> D2 (digital threshold output)
//
// Sends CSV over serial: "<raw>,<digital>\n"
// raw: 0-1023 (0=wet/conductive, 1023=dry)
// digital: 0=wet (below threshold), 1=dry (above threshold)

const int AO_PIN = A0;
const int DO_PIN = 2;
const int BAUD_RATE = 9600;
const int INTERVAL_MS = 1000;

void setup() {
  Serial.begin(BAUD_RATE);
  pinMode(DO_PIN, INPUT);
}

void loop() {
  int raw = analogRead(AO_PIN);
  int dry = digitalRead(DO_PIN);  // HIGH (1) = dry, LOW (0) = wet

  Serial.print(raw);
  Serial.print(",");
  Serial.println(dry);

  delay(INTERVAL_MS);
}
