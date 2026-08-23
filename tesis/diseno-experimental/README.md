# Diseño experimental — batch instrumentado H. illucens

Documento maestro para mandar a construir el batch y ejecutar los
experimentos con protocolos estrictos.

## Contenido
- `main.tex` — documento de diseño (compilar con xelatex)
- `diagrama_reactor.tex` — esquema TikZ (se incluye desde main.tex)

## Compilar
    xelatex main.tex    # dos pasadas si se edita el TOC

## Decisiones de diseño (resumen)
- Aireación forzada de FONDO (placa difusora + plenum): los modelos
  del gemelo mostraron que ventilar la cabecera no controla ni el
  CO2 ni la temperatura dentro de la cama
- Caudal de diseño 0.1–3.8 L/min (0.5–20 ACH), calculado de flujos
  reales medidos (294 mL/h por 750 larvas, escalado a 2000)
- CO2: MH-410D versión 5 %vol (la de 5000 ppm saturó en campaña 1)
- O2: LuminOx LOX-02 (óptico, 0–25 %) → cociente respiratorio RQ
- Baño térmico 30 ± 0.5 °C; cama ≤ 33 °C vía aireación
- Ventana cerrada diaria de 15 min → flujo del día + muestreo GC
- Cosecha por señal: pico CH4/COVs vs caída CO2 vs aplanamiento peso
- Batch nominal: 2000 larvas pesadas, 100 mg/larva/día, cargas
  días 0/3/6, cama 3–4.5 cm, humedad 65–70 %, oscuridad

## Checklist de construcción
- [ ] Placa difusora + plenum
- [ ] Septo en tapa
- [ ] Válvula solenoide modo dual
- [ ] Baño térmico + pilares de peso + celdas de carga
- [ ] Sensor CO2 5 %vol adquirido
- [ ] Sensor O2 LOX-02 adquirido
- [ ] SHT45 + 3x DS18B20 + sensor capacitivo + HX711
- [ ] Bomba + rotámetro + humidificador + filtro
- [ ] Prueba de fugas P-01 aprobada (< 5 %/h)
- [ ] Validación de sensores P-02 (etanol + gas patrón)

## Pendientes del documento
- Completar referencias [2]–[5] desde Zotero (Better BibTeX)
- Decidir rotámetro vs controlador de flujo másico
