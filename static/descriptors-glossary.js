/**
 * Glosario de Descriptores Químicos - Modelo Tox21 (SR-MMP)
 * Actualizado con los descriptores exactos de RDKit devueltos por SHAP.
 */

const GLOSARIO_DESCRIPTORES = {
    // --- TUS DESCRIPTORES ESPECÍFICOS ENCONTRADOS ---
    "MolLogP": "Coeficiente de reparto octanol-agua estimado. Mide la lipofilicidad; valores altos facilitan que el compuesto penetre la membrana mitocondrial.",
    "FractionCSP3": "Fracción de carbonos con hibridación SP3. Mide la 'esfericidad' y complejidad 3D de la molécula frente a estructuras completamente planas.",
    "fr_COO": "Presencia de grupos carboxilato o ácidos carboxílicos ($-COO-$). Afecta directamente el estado de ionización y la carga de la molécula a pH fisiológico.",
    "SMR_VSA10": "Superficie molecular molar refractiva (rango alto). Mide la polarizabilidad y el volumen de zonas específicas de la molécula que interactúan con proteínas.",
    "SMR_VSA3": "Superficie molecular molar refractiva (rango bajo). Evalúa interacciones electrostáticas débiles en parches moleculares específicos.",
    "BCUT2D_MWLOW": "Descriptor de matriz molecular (basado en peso atómico, eigenvalor más bajo). Describe la topología de la molécula enfocándose en las regiones con los átomos más ligeros.",
    "BCUT2D_LOGPHI": "Descriptor de matriz molecular (basado en lipofilicidad, eigenvalor más alto). Informa sobre la distribución de las zonas hidrofóbicas en la superficie de la molécula.",
    "BCUT2D_CHGLO": "Descriptor de matriz molecular (basado en carga parcial, eigenvalor más bajo). Identifica la región con la carga electrónica negativa más concentrada en la estructura.",
    "PEOE_VSA6": "Área de superficie molecular con carga parcial en un rango intermedio (método Gasteiger). Mide cómo se distribuyen las cargas medias en la superficie de la molécula.",
    "PEOE_VSA11": "Área de superficie molecular con cargas fuertemente positivas o negativas. Es crítico para predecir si el compuesto puede alterar el gradiente electroquímico mitocondrial.",
    "VSA_EState3": "Área de superficie combinada con el estado electrotópico (rango 3). Evalúa el potencial de reactividad química y enlaces de hidrógeno de ciertos átomos de la molécula.",
    "SPS": "Puntuación de Proximidad Espacial (Spatial Proximity Score). Mide la complejidad de la forma tridimensional y la conectividad del esqueleto carbonado de la sustancia.",
    "Avglpc": "Carga parcial promedio o índice de polarización calculado en la superficie. Evalúa el potencial general de la molécula para generar interacciones electrostáticas complejas.",

    // --- OTROS DESCRIPTORES COMUNES DE RESPALDO ---
    "MolWt": "Peso molecular total. Influye en la facilidad con la que el compuesto se transporta a través de las barreras biológicas.",
    "TPSA": "Área de superficie polar tópica. Valores bajos facilitan que el compuesto atraviese por difusión pasiva las membranas celulares.",
    "NumAromaticRings": "Número de anillos aromáticos. Estructuras planas que pueden intercalarse en membranas o alterar la cadena respiratoria enzimática.",
    "fr_nitro": "Presencia de grupos nitro ($-NO_2$). Grupo químico reactivo capaz de inducir estrés oxidativo mitocondrial celular."
};

/**
 * Función auxiliar global para buscar definiciones de forma segura.
 * Si el descriptor no existe en la lista, devuelve un mensaje genérico.
 */
function obtenerDescripcionDescriptor(nombreTecnico) {
    if (GLOSARIO_DESCRIPTORES[nombreTecnico]) {
        return GLOSARIO_DESCRIPTORES[nombreTecnico];
    }
    // Mensaje dinámico por si aparece algún descriptor nuevo de RDKit no mapeado
    return `Propiedad fisicoquímica molecular (${nombreTecnico}) detectada por el modelo como factor influyente en la alteración del entorno celular.`;
}