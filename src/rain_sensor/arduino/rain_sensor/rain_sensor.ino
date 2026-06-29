// Rain Sensor - Arduino Sketch
// MH-sensor-series module connected to:
//   AO -> A1 (analog raw value)
//   DO -> D3 (digital threshold output)
//
// Sends CSV over serial: "<raw>,<raining>\n"
// raw: 0-1023 (0=heavy rain/conductive, 1023=dry)
// raining: 1=raining (below threshold), 0=dry (above threshold)

const int AO_PIN = A1;
const int DO_PIN = 3;
const int BAUD_RATE = 9600;
const int INTERVAL_MS = 1000;

void setup() {
  Serial.begin(BAUD_RATE);
  pinMode(DO_PIN, INPUT);
}

void loop() {
  int raw = analogRead(AO_PIN);
  int raining = !digitalRead(DO_PIN);  // invert: 1=raining, 0=dry

  Serial.print(raw);
  Serial.print(",");
  Serial.println(raining);

  delay(INTERVAL_MS);
}
