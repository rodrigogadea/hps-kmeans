module Math.Statistics.KMeans
( euclideanDist
, distanceToCenters
, assignCluster
, cloudCenter
, selectFrom
, kMeans
) where

import qualified Data.Vector as V
import qualified Data.Vector.Unboxed as UV

-- | Euclidean Distance between two points
euclideanDist :: (RealFloat a, UV.Unbox a) => UV.Vector a -> UV.Vector a -> a
euclideanDist x y = sqrt . UV.sum $ UV.zipWith 
                    (\xi yi -> (xi - yi) ^ (2 :: Integer)) x y

-- | Distance from a Point to a set of Centers
distanceToCenters :: (t -> a -> b) -> V.Vector a -> t -> V.Vector b
distanceToCenters distance centers point = V.map (distance point) centers

-- | Assign Points to a Cluster based on the Minimum distance to each Center
assignCluster :: Ord a => V.Vector (V.Vector a) -> V.Vector Int
assignCluster = V.map V.minIndex

-- | Calculates the Center of a Cloud of Points
cloudCenter :: (Fractional a, UV.Unbox a) =>
               V.Vector (UV.Vector a) -> UV.Vector a
cloudCenter cloud = UV.map (/ fromIntegral(V.length cloud)) $ 
                    V.foldl1 (UV.zipWith (+)) cloud

-- | Selects elements of a Vector given its indices
selectFrom :: V.Vector a -> V.Vector Int -> V.Vector a
selectFrom x = V.map (x V.!)

-- | k-Means classifier for a given Distance, Variation Guard and Cloud
kMeans :: (RealFloat a, UV.Unbox a) =>  
          (UV.Vector a -> UV.Vector a -> a) -> a
          -> V.Vector (UV.Vector a) -> V.Vector (UV.Vector a) 
          -> V.Vector (UV.Vector a)
kMeans distance varGuard centers cloud =
       let dists = V.map (distanceToCenters distance centers) cloud 
           assigned = assignCluster dists
           pointAssign = V.map (selectFrom cloud) $ V.fromList 
                         [V.elemIndices x assigned | 
                          x <- [0..(V.length centers - 1)] ]
           newcenters = V.map cloudCenter pointAssign
           variation = V.sum $ V.zipWith distance centers newcenters
       in (if variation > varGuard then 
               kMeans distance varGuard newcenters cloud
           else newcenters)
