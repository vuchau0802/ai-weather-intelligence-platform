# AI Weather Intelligence Platform

A full-stack weather intelligence application that provides real-time weather data, a 7-day forecast, search history management, weather analytics, air quality insights, travel videos, and interactive visualizations using modern web technologies.

---

## Project Overview

This application allows users to search for weather information by entering a city, ZIP/postal code, or landmark. The platform retrieves live weather data from external APIs, stores weather search history in a PostgreSQL database, and presents useful insights through charts, forecast cards, maps, and recommendation panels.

The app is built with a Next.js frontend, an Express/TypeScript backend, Prisma ORM, and PostgreSQL.

---

## Features

### Frontend Features

- Real-time weather search
- 5-day weather forecast
- Fahrenheit temperature display
- Weather condition icons
- Interactive temperature trend chart
- Interactive map view
- Air quality display when enabled
- Smart weather insight messages
- Travel video recommendations
- Search history table
- Export weather history to CSV or JSON
- Delete saved history records

### Backend Features

- RESTful API built with Express and TypeScript
- Weather search endpoint
- CRUD operations for weather search history
- PostgreSQL database persistence with Prisma ORM
- OpenWeather API integration
- OpenCage geocoding integration
- Air quality integration
- YouTube travel video integration
- Export weather history to CSV
- Export weather history to JSON

---

## Steps to run

## Step 1 — Clone the repo
 
```bash
git clone https://github.com/vuchau0802/ai-powered-weather-intelligence-platform.git
cd ai-powered-weather-intelligence-platform
```

## Step 2 — Set up the Backend
 
```bash
cd backend
notepad .env

# Open backend/.env and replace the placeholders
PORT=5000
DATABASE_URL=postgresql://USER:PASSWORD@HOST/DBNAME?sslmode=require
OPENWEATHER_API_KEY=your_key_here
OPENCAGE_API_KEY=your_key_here
YOUTUBE_API_KEY=your_key_here
FRONTEND_URL=http://localhost:3000
ENABLE_GEOCODING=true
ENABLE_AIR_QUALITY=true
ENABLE_YOUTUBE=true

# Install dependencies
npm install

# Generate the Prisma
npm run db:generate

# Create the database table
npm run db:ensure

# Start the backend dev server
npm run dev
# Backend is now running at http://localhost:5000
```

## Step 3 — Set up the Frontend (new terminal)
 
```bash
cd frontend

# Install dependencies
npm install

# Start the frontend dev server
npm run dev

# Frontend is now running at http://localhost:3000
```
