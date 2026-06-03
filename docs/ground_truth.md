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

El modelo YOLOv10s entrenado sobre DocLayNet reconoce las siguientes categorías de regiones dentro de un documento:

| Categoría | Descripción | ¿Presente en nuestros documentos? |
|---|---|---|
| Text | Bloques de texto corrido (párrafos, considerandos) | Sí, muy frecuente |
| Section-header | Encabezados de sección (CONSIDERANDO, RESUELVE, ANEXO) | Sí, frecuente |
| List-item | Elementos de lista numerada o con viñetas | Sí, en articulados y anexos |
| Table | Tablas con datos tabulares | Sí, en algunas ordenanzas |
| Picture | Imágenes, firmas, sellos, escudos | Sí, en varias páginas |
| Title | Título principal del documento | Sí, ocasional |
| Caption | Epígrafes de tablas o figuras | Sí, poco frecuente |
| Page-header | Encabezado de página (membrete) | Sí, en algunas páginas |
| Page-footer | Pie de página (numeración, etc.) | Sí, en varias páginas |
| Formula | Fórmulas matemáticas | No aplica a estos documentos |
| Footnote | Notas al pie | No observado |

## Criterio de referencia para evaluación

En esta etapa inicial del proyecto, la evaluación del modelo se realiza de forma **cualitativa**: se observan visualmente las detecciones sobre las imágenes anotadas y se verifica si las regiones marcadas corresponden efectivamente a las categorías asignadas.

No se cuenta todavía con anotaciones manuales en formato COCO o YOLO que permitan calcular métricas como mAP o IoU de forma automática. Esto queda como parte del trabajo a futuro (ver sección correspondiente en el informe).

Para esta entrega, el ground truth consiste en:

1. **La definición del conjunto de documentos** con su procedencia y características.
2. **La identificación de las categorías esperadas** en función del tipo de contenido de los documentos.
3. **La revisión visual de una muestra representativa** de resultados para verificar la coherencia de las detecciones.

## Observaciones sobre la calidad de los documentos

Algunos aspectos que se detectaron al trabajar con este conjunto:

- Las ordenanzas más extensas (ORD-189, 56 páginas) incluyen contenido muy heterogéneo: texto normativo, tablas presupuestarias, formularios y páginas con firmas escaneadas.
- Las resoluciones tienden a ser más cortas y con estructura más uniforme (texto + firmas).
- Algunas páginas son reversos en blanco de hojas escaneadas a doble faz, lo cual genera páginas sin contenido detectable. El modelo las identifica correctamente con cero detecciones.
- La calidad del escaneo varía entre documentos: algunos tienen buena resolución y otros presentan ligera inclinación o baja nitidez.
