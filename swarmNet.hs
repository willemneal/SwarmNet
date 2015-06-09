import System.Random

randomList :: (Random a) => Int -> [a]
randomList seed = randoms (mkStdGen seed)


sample = randomList 100 :: [Double]

type Rho   = Double
type Theta = Double
type Walker = (Rho, Theta)

getRho u1 u2 | u >1 = 2- u
             | otherwise = u
             where u = u1 + u2

walkers :: [Double]->[Walker]
walkers (u1:u2:u3:xs) = (rho, theta) :  walkers xs
                        where
                          theta = 2*pi*u3
                          rho = getRho u1 u2


getWalkers :: Int -> [Walker]
getWalkers sampleSize = take sampleSize $ walkers sample

-- t = 2*pi*random()
-- u = random()+random()
-- r = if u>1 then 2-u else u
-- [r*cos(t), r*sin(t)]
