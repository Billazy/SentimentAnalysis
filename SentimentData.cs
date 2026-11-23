// <SnippetAddUsings>
// Fournit les attributs [LoadColumn] et [ColumnName] utilisés pour mapper
// les colonnes du fichier (ou de la collection) vers les propriétés C#.
using Microsoft.ML.Data;
// </SnippetAddUsings>

namespace SentimentAnalysis
{
    // <SnippetDeclareTypes>
    /// <summary>
    /// Schéma d'une ligne d'entrée : le texte du commentaire + son étiquette.
    /// </summary>
    public class SentimentData
    {
        // [LoadColumn(0)] : cette propriété est mappée sur la 1re colonne du fichier.
        [LoadColumn(0)]
        public string SentimentText;

        // [LoadColumn(1)] : mappée sur la 2e colonne (0 = négatif, 1 = positif).
        // [ColumnName("Label")] : renomme la colonne en "Label" pour que
        // l'entraîneur sache quelle est la variable cible.
        [LoadColumn(1), ColumnName("Label")]
        public bool Sentiment;
    }

    /// <summary>
    /// Schéma de sortie du modèle : hérite de SentimentData et ajoute
    /// les colonnes produites par la prédiction.
    /// </summary>
    public class SentimentPrediction : SentimentData
    {
        // Étiquette prédite (true = positif, false = négatif).
        // [ColumnName("PredictedLabel")] : nom standard reconnu par l'évaluateur.
        [ColumnName("PredictedLabel")]
        public bool Prediction { get; set; }

        // Probabilité associée à la classe prédite (entre 0 et 1).
        public float Probability { get; set; }

        // Score brut du modèle (logit), avant calibration en probabilité.
        public float Score { get; set; }
    }
    // </SnippetDeclareTypes>
}