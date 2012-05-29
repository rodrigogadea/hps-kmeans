import Math.Statistics.KMeans
import System.Random.MWC as MWC
import System.Random.MWC.Distributions
import Data.Vector as V
import Data.Vector.Unboxed as UV
import Control.Monad.ST()

main :: IO ()
main = do
  -- Initialize the PRNG
  gen <- MWC.initialize (V.singleton 66)

  -- Number of observations and groups
  let cloudSize = 10000 :: Int
      nGroups = 5 :: Int

  -- Simulate multivariate normal variables with Id covariance matrices
  let g1Mean = UV.fromList [4, 0, 0] :: UV.Vector Double ; g1Size = 2000 :: Int
      g2Mean = UV.fromList [0, 4, 0] :: UV.Vector Double ; g2Size = 2000 :: Int
      g3Mean = UV.fromList [0, 0, 4] :: UV.Vector Double ; g3Size = 2000 :: Int
      g4Mean = UV.fromList [0, 4, 4] :: UV.Vector Double ; g4Size = 2000 :: Int
      g5Mean = UV.fromList [4, 4, 4] :: UV.Vector Double ; g5Size = 2000 :: Int
  g1 <- V.replicateM g1Size $ UV.forM g1Mean (\mean -> normal mean 1 gen)
  g2 <- V.replicateM g2Size $ UV.forM g2Mean (\mean -> normal mean 1 gen)
  g3 <- V.replicateM g3Size $ UV.forM g3Mean (\mean -> normal mean 1 gen)
  g4 <- V.replicateM g4Size $ UV.forM g4Mean (\mean -> normal mean 1 gen)
  g5 <- V.replicateM g5Size $ UV.forM g5Mean (\mean -> normal mean 1 gen)
  let myData = g1 V.++ g2 V.++ g3 V.++ g4 V.++ g5

  -- Sample the initial centers
  sampleForCenters <- V.replicateM nGroups $ uniform gen
  let initialCentersIndices = V.map (floor . (*) (fromIntegral cloudSize))
                              (sampleForCenters :: V.Vector Double)
      initialCenters = selectFrom myData initialCentersIndices

  -- Train the algorithm and print the classifier
  print $ kMeans euclideanDist 0.01 initialCenters myData
