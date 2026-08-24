# 01 — Parámetros del modelo

Bioconversión batch de residuos con *Hermetia illucens*.
Núcleo: modelo DEB de [@eriksen2022] + módulo GEI. Flujos en equivalentes de carbono (C-eq).

## A. Núcleo DEB (Eriksen)

| Parámetro | Símbolo | Valor | Rango | Unidad | Nota / condición |
|---|---|---|---|---|---|
| Tasa máx. asimilación específica | $a_{max}$ | 1.30 | 1.1–1.5 | d$^{-1}$ | buenas condiciones; 0.8–1.1 estrés hídrico; 0.25–1.21 con limitación de alimento [@eriksen2022; @eriksen2024] |
| Costo de crecimiento (biomasa estructural) | $Y_B$ | 0.44 | 0.35–0.44 | — | sube con sustrato pobre [@laganaro2021; @eriksen2024] |
| Mantenimiento | $m$ | 0.08 | 0.08–0.13 | d$^{-1}$ | chicken feed / residuo cervecero [@laganaro2021; @eriksen2024] |
| Costo síntesis de lípidos | $Y_L$ | 0.42 | — | — | sustratos granulares [@eriksen2022] |
| Peso máx. biomasa estructural | $B_{max,0}$ | 65 | 65–90 | mg | depende de la cepa [@eriksen2022] |
| Tiempo mín. a prepupa | $t_{p,min}$ | 13 | — | d | condiciones óptimas [@eriksen2022] |
| Caída de $B_{max}$ post-$t_{p,min}$ | $\rho$ | 1.0 | — | mg/d | [@eriksen2022] |
| Razón de peso entre instares | $\delta_I$ | 5 | 4–5 | — | [@eriksen2022] |
| Downregulación asimilación (instar 6) | $\alpha$ | 1.0 | 0.28–3.3 | — | sensible a calidad y humedad del sustrato [@eriksen2022] |
| Downregulación crecimiento → lípidos | $\beta$ | 1.0 | 1–2 | — | reparto de asimilados a reserva [@eriksen2022] |

Indicadores de referencia: $NGE^*_{avg}$ = 0.53–0.58 (buen sustrato), 0.26–0.58 (residuos); $SCE$ = 0.14–0.48 (chicken feed), 0–0.35 en general [@eriksen2024].

## B. Estequiometría — balance de carbono de Eriksen

| Factor | Valor | Uso |
|---|---|---|
| $\delta_{C,B}$ | 0.49 | C en biomasa estructural |
| $\delta_{C,L}$ | 0.74 | C en lípidos |
| $\delta_{C,CO_2}$ | 0.27 | C en CO₂ |
| $\delta_{L,B}$ | ≈0.10 | lípidos estructurales |
| $\delta_O$ | 0.90 | fracción orgánica (SV) de la biomasa estructural |
| $\delta_{DW,B}$ | 0.25 | MS de la biomasa estructural |

Composición larval de referencia (sobre MS): 48 ± 10% proteína, 27 ± 11% lípidos, 14 ± 10% cenizas [@eriksen2022].

Balance por larva (C-eq):

$$r_A = r_B + r_L + r_{CO_2}, \qquad r_{CO_2} = mB + \frac{1-Y_B}{Y_B}\,r_B + \frac{1-Y_L}{Y_L}\,r_L$$

Cierre de batch (criterio de calidad de datos, error < 10–15%):

$$C_{alimento} = C_{larvas} + C_{residuo} + C_{CO_2} + C_{CH_4}$$

## C. Variables controladas — rangos de operación del batch

| Variable | Óptimo | Rango viable | Efecto documentado |
|---|---|---|---|
| Temperatura | 25–30 °C | 25–35 °C | crecimiento y supervivencia [@shumo2019]; el sustrato puede llegar a 45 °C por calor metabólico |
| Humedad sustrato | ≈75% | 45–85% | ↑humedad → ↑CH₄; el CO₂ acumulado sigue la tendencia del peso larval [@chen2019]; larvas más pesadas a 75% [@bekker2021] |
| HR aire | 60–75% | — | <60% deshidrata |
| pH sustrato | 6–8 | sobrevive 2–11 | pH ácido inhibe metanógenos [@pang2020] |
| Ración | ≈100 mg/larva/d | 12.5–200 | mejor conversión [@diener2009] |
| $Q_{10}$ respiración | — | — | **sin valor publicado para BSFL → lo estimamos con nuestros datos (aporte propio)** |

## D. Emisiones GEI de referencia (dimensionar sensores y validar)

| Estudio | Sustrato | CO₂ | CH₄ | N₂O |
|---|---|---|---|---|
| [@ermolaev2019] | residuo alimentario | 96 g/kg residuo (1750 ± 170 g/kg MS larvas) | 49 ± 29 mg/kg MS larvas | 21 ± 13 mg/kg MS larvas |
| [@lindberg2022] | frutas/verduras | 47–147 g/kg peso inicial | CH₄+N₂O: 0.04–1.57 g CO₂-eq/kg | — |
| [@parodi2020] | estiércol porcino | 1956 ± 105 g/kg MS larvas | 10.1 ± 1.33 g/kg MS larvas | GWP 344 ± 43 g CO₂-eq/kg MS larvas |
| [@pang2020] | residuo + paja | 1394 ± 343 g/kg MS larvas | 14 ± 6 mg/kg MS larvas | 7 ± 1 mg/kg MS larvas |
| [@mertenat2019] | biorresiduo | GWP 35 kg CO₂-eq/t (vs 111 compostaje) | 47× menos que compostaje | — |

Hechos clave para el soft-sensor: con larvas el CO₂ casi se duplica vs control sin larvas [@parodi2020] → necesitamos batch de control microbiano. RQ ≈ 1 (carbohidratos), ≈ 0.8 (proteína/grasa) [@parodi2020].

## E. A calibrar con nuestros datos

- $a_{max}$, $Y_B$, $m$ — nuestro sustrato no es chicken feed → esperar valores peores
- $\alpha$, $\beta$ — dependen de sustrato y humedad
- $k_{CH_4}$ y sus moduladores $f(T)$, $f(W)$, $f(pH)$
- $Q_{10}$ — no existe en literatura para BSFL

## Referencias (citekeys provisionales → sincronizar con Zotero)

- [@eriksen2022] Eriksen NT (2022) Dynamic modelling of feed assimilation, growth, lipid accumulation, and CO2 production in black soldier fly larvae. *PLoS ONE* 17(10): e0276605.
- [@eriksen2024] Eriksen NT (2024) Metabolic performance and feed efficiency of black soldier fly larvae. *Front. Bioeng. Biotechnol.* 12:1397108.
- [@laganaro2021] Laganaro M, Bahrndorff S, Eriksen NT (2021) Growth and metabolic performance of black soldier fly larvae grown on low and high-quality substrates. *Waste Manag.* 121: 198–205.
- [@bekker2021] Bekker NS et al. (2021) Impact of substrate moisture content on growth and metabolic performance of black soldier fly larvae. *Waste Manag.* 127: 73–79.
- [@padmanabha2020] Padmanabha M, Kobelski A, Hempel A-J, Streif S (2020) A comprehensive dynamic growth and development model of Hermetia illucens larvae. *PLoS ONE* 15: e0239084.
- [@ermolaev2019] Ermolaev E, Lalander C, Vinnerås B (2019) Greenhouse gas emissions from small-scale fly larvae composting with Hermetia illucens. *Waste Manag.* 96: 65–74.
- [@mertenat2019] Mertenat A, Diener S, Zurbrügg C (2019) Black Soldier Fly biowaste treatment — Assessment of global warming potential. *Waste Manag.* 84: 173–181.
- [@parodi2020] Parodi A et al. (2020) Bioconversion efficiencies, greenhouse gas and ammonia emissions during black soldier fly rearing — A mass balance approach. *J. Clean. Prod.* 271: 122488.
- [@pang2020] Pang W et al. (2020) Reducing greenhouse gas emissions and enhancing carbon and nitrogen conversion in food wastes by the black soldier fly. *J. Environ. Manage.* 260: 110066.
- [@chen2019] Chen J et al. (2019) Effect of moisture content on greenhouse gas and NH3 emissions from pig manure converted by black soldier fly. *Sci. Total Environ.* 697: 133840.
- [@lindberg2022] Lindberg L et al. (2022) Process efficiency and greenhouse gas emissions in black soldier fly larvae composting of fruit and vegetable waste. *J. Clean. Prod.*
- [@shumo2019] Shumo M et al. (2019) Influence of temperature on selected life-history traits of black soldier fly reared on two common urban organic waste streams in Kenya. *Animals* 9: 79.
- [@diener2009] Diener S, Zurbrügg C, Tockner K (2009) Conversion of organic material by black soldier fly larvae: establishing optimal feeding rates. *Waste Manag. Res.* 27: 603–610.
