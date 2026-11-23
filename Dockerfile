# Étape 1 : Build
FROM mcr.microsoft.com/dotnet/sdk:10.0 AS build
WORKDIR /src

# Copier les fichiers projet pour restaurer les dépendances
COPY ["SentimentAnalysis.csproj", "."]
COPY ["SentimentAnalysis.Api/SentimentAnalysis.Api.csproj", "SentimentAnalysis.Api/"]
RUN dotnet restore "SentimentAnalysis.Api/SentimentAnalysis.Api.csproj"

# Copier le reste du code
COPY . .
WORKDIR "/src/SentimentAnalysis.Api"
RUN dotnet publish "SentimentAnalysis.Api.csproj" -c Release -o /app/publish

# Étape 2 : Runtime
FROM mcr.microsoft.com/dotnet/aspnet:10.0 AS runtime
WORKDIR /app
COPY --from=build /app/publish .
COPY --from=build /src/SentimentAnalysis.Api/sentiment_model.zip .

ENV ASPNETCORE_URLS=http://+:8080
EXPOSE 8080

ENTRYPOINT ["dotnet", "SentimentAnalysis.Api.dll"]