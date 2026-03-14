-- Select all columns
SELECT * FROM movies;

-- Select specific columns
SELECT title, director FROM movies;

-- Select with condition
SELECT title FROM movies WHERE year = 2010;

-- Select distinct values
SELECT DISTINCT director FROM movies;
