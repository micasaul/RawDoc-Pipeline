# Definición del Conjunto de Referencia (Ground Truth)

## Origen de los documentos

Los documentos utilizados en este proyecto provienen del sitio oficial de la Universidad Autónoma de Entre Ríos (UADER). Se trata de actos administrativos emitidos por el Consejo Superior, descargados en formato PDF desde el portal institucional.

Se seleccionaron dos tipos de documentos:
- **Ordenanzas del Consejo Superior** (prefijo ORD-CS): normativas de alcance general.
- **Resoluciones del Consejo Superior** (prefijo Res-CS): disposiciones sobre temas puntuales.

Estos documentos representan un caso de uso real y concreto: digitalizar y estructurar la información contenida en actos administrativos universitarios que, al día de hoy, solo están disponibles como archivos PDF escaneados o generados de forma mixta (texto digital con firmas y sellos en imagen).

## Composición del conjunto

| Documento | Tipo | Páginas |
|---|---|---|
| ORD-185 | Ordenanza | 8 |
| ORD-CS-186-25 | Ordenanza | 20 |
| ORD-CS-N°-187 | Ordenanza | 24 |
| ORD-CS-N°-188-25 | Ordenanza | 20 |
| ORD-CS-N°-189-comprimido | Ordenanza | 56 |
| Res-CS-054-25-27-03-2025 | Resolución | 2 |
| Res-CS-055-25-27-03-2025 | Resolución | 8 |
| Res-CS-056-25-27-03-2025 | Resolución | 2 |
| Res-CS-073-25-27-03-2025 | Resolución | 4 |
| Res-CS-078-25-27-03-2025 | Resolución | 4 |

**Muestra de Validación Manual (Ground Truth): 10 documentos, 148 páginas.**  
*(El corpus extendido total procesado y evaluado por el pipeline asciende a 20 documentos y 250 páginas, detallado en la Sección 5.1 del Informe Técnico).*

El conjunto incluye variedad de estructuras: páginas con texto corrido, listas numeradas, tablas de datos, formularios, páginas con firmas y sellos, e incluso reversos de hojas escaneadas. Esta diversidad es útil para evaluar la robustez del modelo de detección de layout.

## Categorías de layout esperadas

| Categoría YOLO | Descripción | ¿Presente en UADER? | Columna en Excel de Validación |
|---|---|---|---|
| **Text** | Bloques de texto corrido (párrafos, considerandos) | Sí, muy frecuente | `TEXTO` |
| **Section-header** | Encabezados de sección (CONSIDERANDO, RESUELVE, ANEXO) | Sí, frecuente | `SUBTITULO (SH)` |
| **List-item** | Elementos de lista numerada o con viñetas | Sí, frecuente | `LISTA (ITEMS)` |
| **Table** | Tablas con datos tabulares | Sí, ocasional | `TABLA` |
| **Picture** | Imágenes, firmas, sellos, escudos institucionales | Sí, frecuente | `IMAGEN` (Fusionado con "Sellos") |
| **Title** | Título principal del documento | Sí, ocasional | `TEXTO` (o `SUBTITULO (SH)`) |
| **Caption** | Epígrafes de tablas o figuras | Sí, poco frecuente | `TEXTO` |
| **Page-header** | Encabezado de página (membrete) | Sí, frecuente | `P. HEADER` |
| **Page-footer** | Pie de página (numeración, etc.) | Sí, frecuente | `P. FOOTER` |
| **Formula** | Fórmulas matemáticas | No aplica | *(Ignorar)* |
| **Footnote** | Notas al pie | No aplica | *(Ignorar)* |
| *(Sin detección)* | Páginas sin contenido detectable / reversos | Sí, ocasional | `HOJA EN BLANCO` |

## Criterio de referencia y Validación Manual (Ground Truth)

Para validar cuantitativamente el desempeño del pipeline, el equipo construyó una planilla de Ground Truth y control de experimentos, cuya versión congelada e inmutable se encuentra preservada en el repositorio en **[docs/experimento_confianza.xlsx](experimento_confianza.xlsx)** (respaldada además en el enlace original de **[Google Sheets](https://docs.google.com/spreadsheets/d/1auYnDuoBU9F2DGiNMLovTlKGoXOYFVR43KZiilfujnU/edit?usp=sharing)**). Esta planilla cuenta con dos pestañas de registro:
1. `Hoja 1` (Fase Baseline): Validación manual de Ground Truth sobre las primeras 10 muestras contrastadas contra el modelo baseline inicial (YOLOv10s).
2. `Corpus_20_Docs_OCR` (Fase Final): Métricas consolidadas del procesamiento batch completo sobre los 20 documentos (250 páginas) con el pipeline definitivo (YOLO11m + Tesseract 5).

Desde una perspectiva de ingeniería y análisis de dominio normativo, la validación no requiere anotar exhaustivamente las 11 categorías nativas del dataset DocLayNet (muchas de las cuales, como fórmulas matemáticas o notas al pie complejas, no se encuentran en este tipo de actos administrativos). En su lugar, el diseño se enfoca prioritariamente en aquellas categorías **determinantes para el pipeline de OCR selectivo**:

1. **Tablas (`Table`)**: Bloques de datos tabulares que requieren un procesamiento de extracción estructurado especial o exclusión del OCR lineal.
2. **Imágenes, Firmas y Sellos (`Picture`)**: Regiones que contienen elementos gráficos no textuales (como firmas manuscritas, sellos oficiales o escudos institucionales).
3. **Páginas sin elementos detectables por encima del umbral**: Identificadas mediante la ausencia total de detecciones (0 bboxes), esenciales para omitir procesamiento innecesario en el OCR.
4. **Zonas de texto a extraer**: Agrupadas de manera general (principalmente a través de `Text`, `Section-header` y `List-item`).

### Definición Formal de Métricas de Evaluación (TP, FP, FN, Precisión, Recall y F1-Score)
Para contrastar cuantitativamente las predicciones del modelo frente al Ground Truth manual, se definen formalmente las siguientes condiciones de evaluación por clase y página:

* **Verdadero Positivo (TP):** Elemento documentado en el Ground Truth manual que fue correctamente detectado por el modelo en la página correspondiente, coincidiendo en clasificación semántica y delimitación espacial.
* **Falso Positivo (FP):** Detección reportada por el modelo sobre una región donde no existe tal elemento en el Ground Truth (por ejemplo, detectar ruido de escáner como texto), o clasificar erróneamente un tipo de bloque.
* **Falso Negativo (FN):** Elemento real existente en el Ground Truth manual que fue omitido por el modelo por quedar su puntuación por debajo del umbral de confianza seleccionado ($\text{conf} < 0.20$).

A partir de estos valores se calculan las métricas estadísticas estándar de detección:

$$\text{Precisión} = \frac{\text{TP}}{\text{TP} + \text{FP}}$$

$$\text{Recall (Sensibilidad)} = \frac{\text{TP}}{\text{TP} + \text{FN}}$$

$$F_1\text{-Score} = 2 \cdot \frac{\text{Precisión} \cdot \text{Recall}}{\text{Precisión} + \text{Recall}}$$

> **Distinción Conceptual Metodológica: Confianza vs. Precisión**  
> Es fundamental distinguir ambos conceptos para no incurrir en confusiones técnicas:
> * La **Confianza** (*Confidence Score*, $\in [0, 1]$) es un valor de probabilidad interna que el modelo asigna a cada predicción (proveniente de la función Softmax de la red neuronal). Indica qué tan segura está la red de su salida.
> * La **Precisión** (*Precision*, $\in [0, 1]$) es una métrica de rendimiento externo calculada a posteriori: indica qué porcentaje de las detecciones realizadas por el modelo fueron verdaderamente correctas al contrastarse contra el Ground Truth humano.
> Una confianza alta no garantiza infalibilidad; por ende, se evalúa la precisión empírica mediante la matriz de confusión.

## Observaciones sobre la calidad de los documentos

Algunos aspectos que se detectaron al trabajar con este conjunto:

- Las ordenanzas más extensas (ORD-189, 56 páginas) incluyen contenido muy heterogéneo: texto normativo, tablas presupuestarias, formularios y páginas con firmas escaneadas.
- Las resoluciones tienden a ser más cortas y con estructura más uniforme (texto + firmas).
- Algunas páginas son reversos en blanco de hojas escaneadas a doble faz, lo cual genera páginas sin contenido detectable. El modelo las identifica correctamente con cero detecciones.
- La calidad del escaneo varía entre documentos: algunos tienen buena resolución y otros presentan ligera inclinación o baja nitidez.
