using Microsoft.Extensions.ML;
using SentimentAnalysis;
using System.Text.Json;

var builder = WebApplication.CreateBuilder(args);

// Configurer la désérialisation JSON :
// - PropertyNameCaseInsensitive : ignore la casse (sentimentText vs SentimentText)
// - IncludeFields : inclut les champs publics (SentimentText est un field)
builder.Services.ConfigureHttpJsonOptions(options =>
{
    options.SerializerOptions.PropertyNameCaseInsensitive = true;
    options.SerializerOptions.IncludeFields = true;
});

builder.Services.AddPredictionEnginePool<SentimentData, SentimentPrediction>()
    .FromFile(
        modelName: "SentimentModel",
        filePath: "sentiment_model.zip",
        watchForChanges: true
    );

var app = builder.Build();

app.MapGet("/", () => Results.Ok(new
{
    message = "Sentiment Analysis API",
    endpoints = new[] { "POST /predict", "GET /health" }
}));

app.MapPost("/predict", (SentimentData input, PredictionEnginePool<SentimentData, SentimentPrediction> pool) =>
{
    var prediction = pool.Predict("SentimentModel", input);

    return Results.Ok(new
    {
        text = input.SentimentText,
        prediction = prediction.Prediction ? "Positive" : "Negative",
        probability = prediction.Probability,
        score = prediction.Score
    });
});

app.Run();