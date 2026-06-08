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

**Total: 10 documentos, 148 páginas.**

El conjunto incluye variedad de estructuras: páginas con texto corrido, listas numeradas, tablas de datos, formularios, páginas con firmas y sellos, e incluso páginas en blanco (reversos de hojas escaneadas). Esta diversidad es útil para evaluar la robustez del modelo de detección de layout.

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
| *(Sin detección)* | Páginas vacías / reversos | Sí, ocasional | `HOJA EN BLANCO` |

## Criterio de referencia y Validación Manual (Ground Truth)

Para validar cuantitativamente el desempeño de YOLOv10s, el equipo construyó una planilla de Ground Truth manual (denominada **[ExperimentoConfianza](https://docs.google.com/spreadsheets/d/1auYnDuoBU9F2DGiNMLovTlKGoXOYFVR43KZiilfujnU/edit?usp=sharing)**). Siguiendo las directrices del docente orientador, la validación no requiere anotar exhaustivamente las 11 categorías nativas del modelo, sino enfocarse prioritariamente en aquellas **críticas para el pipeline de OCR**:

1. **Tablas (`Table`)**: Bloques de datos tabulares que requieren un procesamiento de extracción estructurado especial o exclusión del OCR lineal.
2. **Imágenes, Firmas y Sellos (`Picture`)**: Regiones que contienen elementos gráficos no textuales (como firmas manuscritas, sellos oficiales o escudos institucionales).
3. **Páginas en blanco**: Identificadas mediante la ausencia total de detecciones (0 bboxes), esenciales para omitir procesamiento innecesario en el OCR.
4. **Zonas de texto a extraer**: Agrupadas de manera general (principalmente a través de `Text`, `Section-header` y `List-item`).

### Simplificación y Mapeo en la Planilla de Validación
Para evitar redundancias operativas, se unificaron las detecciones manuales bajo la misma taxonomía del modelo:
* **Fusión de Firmas y Sellos en `Picture`**: Inicialmente se planteó registrar las firmas y sellos por separado. No obstante, dado que el modelo preentrenado en DocLayNet detecta firmas, logos y sellos bajo la categoría genérica `Picture`, registrar por separado estas clases en el Excel manual resultaría redundante. Por ende, en la planilla manual se consolidan bajo la columna `Picture` (que representa cualquier elemento gráfico a ignorar por el OCR).
* **Métricas evaluadas**: Se contrastan las detecciones del script `analisis_resultados.py` frente al conteo manual por página para calcular falsos positivos, verdaderos positivos y F1-Score en diferentes umbrales (0.15, 0.20, 0.25). Esto permite calibrar el clasificador para el inicio de la etapa de extracción de texto.

## Observaciones sobre la calidad de los documentos

Algunos aspectos que se detectaron al trabajar con este conjunto:

- Las ordenanzas más extensas (ORD-189, 56 páginas) incluyen contenido muy heterogéneo: texto normativo, tablas presupuestarias, formularios y páginas con firmas escaneadas.
- Las resoluciones tienden a ser más cortas y con estructura más uniforme (texto + firmas).
- Algunas páginas son reversos en blanco de hojas escaneadas a doble faz, lo cual genera páginas sin contenido detectable. El modelo las identifica correctamente con cero detecciones.
- La calidad del escaneo varía entre documentos: algunos tienen buena resolución y otros presentan ligera inclinación o baja nitidez.
