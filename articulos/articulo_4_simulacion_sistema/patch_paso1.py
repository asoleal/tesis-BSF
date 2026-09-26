# patch_paso1.py — reorientacion A1: titulo + resumen + introduccion
from pathlib import Path

# --- titulo ---
m = Path("main_art4.tex")
t = m.read_text()
old_t = "\\title{\\textbf{Simulacion del Sistema de Bioconversion Instrumentado: Gemelo Digital 0D y Verificacion CFD de la Camara}}"
new_t = "\\title{\\textbf{Sistema de Medicion de CO$_2$ en Cria de \\emph{Hermetia illucens}: Cierres Adaptativos, Verificacion CFD de la Mezcla y Gemelo Digital}}"
if t.count(old_t) == 1:
    t = t.replace(old_t, new_t)
m.write_text(t)

# --- resumen + introduccion ---
f = Path("contenido_art4.tex")
s = f.read_text()
ini = "\\begin{abstract}"
fin = "\\section{Modelo del sistema}"
assert s.count(ini) == 1 and s.count(fin) == 1
i0, i1 = s.index(ini), s.index(fin)

nuevo = r'''\begin{abstract}
Presentamos un sistema de medicion de tasas de emision de CO$_2$ para cria
intensiva de \emph{Hermetia illucens} en contenedor, orientado a resolver
dos problemas de la instrumentacion de bajo costo: la saturacion del sensor
NDIR a lo largo del ciclo ---incluido el ayuno de la prepupa--- y la
heterogeneidad espacial del aire, que invalida la pendiente de acumulacion
como estimador de la tasa. El sistema combina tres elementos: (i) un
calendario de cierres adaptativo de dos pasadas, con duraciones entre 2 y
30 min fijadas por el margen del sensor, que mantiene la concentracion
dentro del rango en los escenarios extremos (picos de 4978 y 4604 ppm
frente al tope de 5000); (ii) la verificacion por dinamica de fluidos
computacional del supuesto de mezcla homogenea, que no se sostiene en fase
ventilada ($\tau_{mix} > 720$ s frente a $\tau_{aire} \approx 175$ s) y se
recupera con un ventilador pequeno en la tapa de la camara sellada
($\tau_{mix} = 4$--$24$ s, margen de 10--30$\times$ sobre los cierres); y
(iii) un gemelo digital de dimension cero alimentado con las tasas medidas,
que reproduce el orden de magnitud del ciclo observado en un experimento
preliminar de cria (panera de 12.4 L, 700 larvas). El sistema queda listo
para la campana de factores de emision.

\textbf{Palabras clave:} bioconversion, \emph{Hermetia illucens}, gases de
efecto invernadero, camara de emision, gemelo digital, dinamica de fluidos
computacional.
\end{abstract}

\section{Introduccion}
La bioconversion de residuos organicos con larvas de la mosca soldado negra
(\emph{Hermetia illucens}, BSF) transforma sustratos de bajo valor en biomasa
rica en proteina y lipidos ---utilizable como ingrediente acuicola y
avicola--- y en un residuo carbonizado (frass) con valor fertilizante
\parencite{goldBiowasteTreatmentBlack2020,surendraBioconversionOrganicWastes2016,
tomberlinBugBookBlackSoldier2025}. Frente al compostaje tradicional, el proceso
reduce el volumen del residuo en dias y opera a escala de contenedor, lo que lo
hace apto para sistemas intensivos instrumentados.

La intensificacion del proceso hace necesario cuantificar sus emisiones de
gases de efecto invernadero (GEI): CO$_2$ biogenico, producto de la
respiracion larvaria y microbiana, y CH$_4$, que aparece cuando zonas del
lecho se vuelven anoxicas
\parencite{boakye-yiadomGreenhouseGasEmissions2022,
ermolaevGreenhouseGasEmissions2019,parodiBioconversionEfficienciesGreenhouse2020a,
filonchykGreenhouseGasesEmissions2024a}. La importancia del reporte de estas
emisiones esta dada por los compromisos de inventario nacional de GEI
\parencite{intergovernmentalpanelonclimatechangeipccClimateChange20212023}.
En este trabajo la medicion se realiza dentro de la propia camara de cria,
acondicionada como camara de emision: la ventilacion conmuta entre una fase
abierta, con renovacion de aire a caudal controlado, y cierres estaticos en
los que la concentracion acumula la tasa de emision
\parencite{bertinMeasuringMethaneEmissions2026}. La instrumentacion de bajo
costo ---NDIR para CO$_2$, semiconductores para CH$_4$ y sensores de
O$_2$--- hace viable el monitoreo continuo
\parencite{dubeyLowCostCO2NDIR2024,kiplimoAddressingLowCostMethane2024,
mitchellCalibrationLowCostMethane2024}, pero impone dos condiciones que el
diseno debe garantizar. La primera es de rango: el sensor tiene un tope de
concentracion y la tasa del ciclo varia desde un maximo en el crecimiento
activo hasta casi cero en el ayuno de la prepupa, de modo que la duracion
del cierre debe adaptarse para no saturar el sensor ni perder resolucion.
La segunda es espacial: el metodo de la pendiente exige que el aire interno
sea homogeneo (supuesto A1); si la camara tiene zonas muertas, la lectura no
representa el estado del sistema y la tasa estimada queda sesgada.

El modelado de la bioconversion con BSF cuenta con antecedentes solidos en
la dinamica energetica de los organismos (DEB):
\textcite{eriksenDynamicModellingFeed2022} propuso un modelo dinamico
calibrado que predice crecimiento, acumulacion de lipidos y emision de
CO$_2$ larvario, refinado despues con datos metabolicos adicionales
\parencite{eriksenMetabolicPerformanceFeed2024}; otros trabajos han
extendido el modelado matematico y computacional del sistema larva--sustrato
\parencite{katchaliMathematicalComputationalModeling2025,
grausaMetabolicModelingHermetia2023}. En paralelo, los gemelos digitales
ganan terreno como herramienta de supervision y prediccion en sistemas
biologicos intensivos \parencite{luReviewIntelligentGreenhouse2025a}. Un
gemelo de emisiones, sin embargo, no es mejor que las tasas que lo
alimentan: requiere un instrumento que las mida de forma defendible durante
todo el ciclo. Hasta donde alcanza nuestro conocimiento, la literatura no
reporta una verificacion del supuesto de mezcla homogenea en camaras de
cria instrumentadas, ni un diseno de mezcla forzada orientado a la medicion
de emisiones en este tipo de sistema, ni garantias de rango del sensor para
ciclos completos que incluyan el ayuno de la prepupa.

El objetivo de este articulo es disenar y verificar, antes de la campana
experimental, un sistema de medicion de tasas de CO$_2$ para cria
intensiva de BSF en contenedor, que mantenga al sensor dentro de su rango
durante todo el ciclo y garantice la homogeneidad del aire durante los
cierres. Tres preguntas ordenan el trabajo: (P1)~de rango: ?`un calendario
de cierres de duracion adaptativa mantiene al NDIR dentro de su rango
durante todo el ciclo, desde el pico de emision del crecimiento activo
hasta el ayuno de la prepupa? (P2)~de mezcla: ?`el supuesto de mezcla
homogenea (A1) se sostiene en la fase ventilada y en la fase de cierre, y
que intervencion de diseno lo garantiza cuando falla? (P3)~de integracion:
?`las tasas medidas alimentan un gemelo digital de dimension cero que
reproduce, en orden de magnitud, el ciclo de emision observado en cria
real? Las contribuciones son: (i) el calendario de cierres adaptativo de
dos pasadas con cotas operativas, que cumple el tope del sensor en los
escenarios extremos; (ii) el veredicto cuantitativo del CFD sobre A1 y el
diseno de un ventilador interno con ley de escalamiento verificada; y
(iii) la integracion con un gemelo digital 0D y su demostracion con datos
de cria real. La Seccion 2 describe el sistema experimental y los modelos;
la Seccion 3 presenta los resultados por pregunta; la Seccion 4 discute las
limitaciones y el trabajo futuro, incluida una capa de correccion con redes
neuronales informadas por la fisica (PINN) en entrenamiento piloto. La
Figura~\ref{fig:esquema} muestra el sistema.

\begin{figure}[htbp]
  \centering
  \includegraphics[width=0.9\textwidth]{imagenes/esquema_sistema.pdf}
  \caption{Camara de cria instrumentada como camara de emision: lecho de
  cria, ventilador en tapa, puertos de entrada y salida con caudal
  controlado, sensores de CO$_2$, CH$_4$, O$_2$, temperatura/humedad y
  presion, y el gemelo digital que procesa las mediciones.}
  \label{fig:esquema}
\end{figure}

'''

s = s[:i0] + nuevo + s[i1:]
f.write_text(s)
print("OK: paso 1 — titulo, resumen e introduccion reorientados")
