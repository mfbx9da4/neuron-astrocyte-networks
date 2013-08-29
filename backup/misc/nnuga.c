/*-----------------------------------------------------------------------*\
|                                                                         |
|                  Robotics 95  --  Final Course Project                  |
|                                                                         |
|                    By :  Ziv Pollack  &  Omri Weisman                   |
|                                                                         |
|               NNUGA - Neural Network Using Genetic Algorithms           |
|                                                                         |
\*-----------------------------------------------------------------------*/

/*
 * File name : nnuga.c
 *
 * This program is an implementation of a Neural Network (NN) which learns using
 * Genetic Algorithms (GA). It runs from a Tk shell.
 *
 * It reads points from a file and creates an output file which describes
 * points that correspond to lines that seperate the plain into several
 * regions, regions where the NN's output will be true, and a regions
 * where the NN's output will be false.
 *
 */

#include <stdio.h>
#include <math.h>
#include <sys/time.h>


/* NN related */
#define NUM 2         /* Number of input nodes */
#define LIMIT 150     /* Maximum number of inputs the system can handle */
#define SESSIONS 500  /* Number of training sessions that we'll put the system through */

/* GA related */
#define POPS    10    /* Number of populations */
#define SIZE    25    /* Size of vector in the genetic algorithms */
#define MAXPOP  60    /* Size of population */
#define BESTPOP 4     /* Number of individuals taken from the best */
#define SELPOP  8     /* SELPOP-BESTPOP = Number of people selected randomly on each gen. */
#define NEWPOP  18    /* NEWPOP-SELPOP = Number of new people, created randomly on each gen. */
#define MUT1    25    /* MUT1-NEWPOP = Number of mutations in the first mutation group */
#define MIXGEN  10    /* Number of generations between population mixing */

typedef struct
{
  float p[NUM];
} vector;


/* NN related */
vector test[LIMIT], w1, w2, w3, w4, w5, w6;
int hits[LIMIT], total;
float w7[6];
int b1, b2, b3, b4, b5, b6, b7;

/* GA related */
float pop[POPS][MAXPOP][SIZE];
int score[POPS][MAXPOP];


/*-----------------------------------------------------------------------*\
|                                                                         |
|  Randomize                                                              |
|                                                                         |
\*-----------------------------------------------------------------------*/
randomize()
{
  struct timeval tp;
  struct timezone tzp;

  /* Use time of day to feed the random number generator seed */
  gettimeofday( &tp, &tzp);
  srandom( tp.tv_sec );
}


/*-----------------------------------------------------------------------*\
|                                                                         |
|  irand( range )  - return a random integer in the range 0..(range-1)    |
|                                                                         |
\*-----------------------------------------------------------------------*/
int irand( range )
     int range;
{
  return( random() % range );
}


/*-----------------------------------------------------------------------*\
|                                                                         |
|  scalar_mult - multiply two vectors                                     |
|                                                                         |
\*-----------------------------------------------------------------------*/
float scalar_mult( x, y ) vector x, y;
{
  int i;
  float s = 0.0;
  for ( i = 0 ; i < NUM ; i++ )  s += ( x.p[i] * y.p[i] );
  return s;
}


/*-----------------------------------------------------------------------*\
|                                                                         |
|  This function computes the NN's output for a certain input vector.     |
|  The NN is constructed from 2 layers, first layer has 6 neurons,        |
|  second layer has 1 neuron.                                             |
|                                                                         |
\*-----------------------------------------------------------------------*/
int net( x ) vector x;
{
  /* First layer */
  float a1 = atanpi( scalar_mult( w1, x ) + b1 ) / 1.6;  /* atan transfer function */
  float a2 = atanpi( scalar_mult( w2, x ) + b2 ) / 1.6;  /* atan transfer function */
  int   a3 = ( scalar_mult( w3, x ) + b3 ) > 0;          /* hardlim transfer function */
  int   a4 = ( scalar_mult( w4, x ) + b4 ) > 0;          /* hardlim transfer function */
  float a5 = scalar_mult( w5, x ) + b5 ;                 /* linear transfer function */
  float a6 = scalar_mult( w6, x ) + b6 ;                 /* linear transfer function */
  
  /* Second layer */
  float a7 = ( a1*w7[0] + a2*w7[1] + a3*w7[2] + a4*w7[3] +
               a5*w7[4] + a6*w7[5] + b7 ) > 0.0;         /* hardlim transfer function */
  
  return(a7);
}


/*-----------------------------------------------------------------------*\
|                                                                         |
|  pop_swap( p, a, b )  -  swap two vectors and scores in the population p|
|                                                                         |
\*-----------------------------------------------------------------------*/
pop_swap( p, a, b )
     int p, a, b;
{
  int t, i;

  /* Swap vector */
  for ( i = 0 ; i < SIZE ; i++ )
    {
      t = pop[p][a][i];
      pop[p][a][i] = pop[p][b][i];
      pop[p][b][i] = t;
    }

  /* Swap score */
  t = score[p][a];
  score[p][a] = score[p][b];
  score[p][b] = t;
}


/*-----------------------------------------------------------------------*\
|                                                                         |
|  apply( p, i ) -  apply the i vector of the population p on the NN      |
|                                                                         |
\*-----------------------------------------------------------------------*/
apply( p, i )
     int p, i;
{
  /* Get the weights and biases of the neurons from the GA vector */
  w1.p[0] = pop[p][i][0];  w1.p[1] = pop[p][i][1];  b1 = pop[p][i][2];
  w2.p[0] = pop[p][i][3];  w2.p[1] = pop[p][i][4];  b2 = pop[p][i][5];
  w3.p[0] = pop[p][i][6];  w3.p[1] = pop[p][i][7];  b3 = pop[p][i][8];
  w4.p[0] = pop[p][i][9];  w4.p[1] = pop[p][i][10];  b4 = pop[p][i][11];
  w5.p[0] = pop[p][i][12];  w5.p[1] = pop[p][i][13];  b5 = pop[p][i][14];
  w6.p[0] = pop[p][i][15];  w6.p[1] = pop[p][i][16];  b6 = pop[p][i][17];
  
  w7[0] = pop[p][i][18];
  w7[1] = pop[p][i][19];
  w7[2] = pop[p][i][20];
  w7[3] = pop[p][i][21];
  w7[4] = pop[p][i][22];
  w7[5] = pop[p][i][23];
  b7    = pop[p][i][24];
}


/*-----------------------------------------------------------------------*\
|                                                                         |
|  pop_copy( p1, a, p2, b ) - copy the vector b in the population p2 into |
|                             the vector a in the population p1.          |
|                                                                         |
\*-----------------------------------------------------------------------*/
pop_copy( p1, a, p2, b)
     int p1, a, p2, b;
{
  int i;

  for ( i = 0 ; i < SIZE ; i++ )
    pop[p1][a][i] = pop[p2][b][i];
}


/*-----------------------------------------------------------------------*\
|                                                                         |
|  Initialize the populations                                             |
|                                                                         |
\*-----------------------------------------------------------------------*/
make_initial_population()
{
  int p, i, j;

  for ( p = 0 ; p < POPS ; p++ )
    {
      /* Half population gets values from -1 to 1 */
      for ( i = 0 ; i < (MAXPOP/2) ; i++ )
        for ( j = 0 ; j < SIZE ; j++ )
          pop[p][i][j] = ((random()&1048575) / 1000000.0 - 0.5) * 2;

      /* Half population gets values from -100 to 100 */
      for ( i = (MAXPOP/2) ; i < MAXPOP ; i++ )
        for ( j = 0 ; j < SIZE ; j++ )
          pop[p][i][j] = ((random()&1048575) / 10000.0 - 50) * 2;
    }
}


/*-----------------------------------------------------------------------*\
|                                                                         |
|   Calculate the scores of all the vectors in all the populations        |
|                                                                         |
\*-----------------------------------------------------------------------*/
calc_score()
{
  int p, i;

  for ( p = 0 ; p < POPS ; p++ )
    for ( i = 0 ; i < MAXPOP ; i++ )
      {
        apply( p, i );
        score[p][i] = check_performance();
      }
}


/*-----------------------------------------------------------------------*\
|                                                                         |
|   Sort the populations                                                  |
|                                                                         |
\*-----------------------------------------------------------------------*/
sort_population()
{
  int p, i, j, k, best;

  /* Use insert sort */
  for ( p = 0 ; p < POPS ; p++ )
    for ( i = 0 ; i < (MAXPOP-1) ; i++ )
      {
        best = score[p][i];
        for ( j = (i+1) ; j < MAXPOP ; j++ )
          if ( score[p][j] > best )
            {
              best = score[p][j];
              k = j;
            }
        if ( best > score[p][i] )
          pop_swap( p, i, k );
      }
}


/*-----------------------------------------------------------------------*\
|                                                                         |
|   Show (on the standard output) the best scores of all populations      |
|                                                                         |
\*-----------------------------------------------------------------------*/
statistics( generation )
     int generation;
{
  int p;

  if ( generation % MIXGEN == 0 )
    printf("-----------------------------\n");
  printf(" %4d) First are: ", generation);
  for ( p = 0 ; p < POPS ; p++ )  printf("%3d ", score[p][0] );
  printf(" (from %d)\n",total);
}


/*-----------------------------------------------------------------------*\
|                                                                         |
|  Generate the next generation in all populations                        |
|                                                                         |
\*-----------------------------------------------------------------------*/
make_next_generation( generation )
     int generation;
{
  int p, i, j, k1, k2, m;
  float dev;
  
  for ( p = 0 ; p < POPS ; p++ )
    {
      /* keep best - BESTPOP */
      /* add another group, randomly  - (SELPOP-BESTPOP) */
      for ( i = BESTPOP ; i < SELPOP ; i++ )
        pop_swap( p, i, (irand( MAXPOP - i ) + i) );
      
      /* create new individuals */
      for ( i = SELPOP ; i < NEWPOP ; i++ )
        for ( j = 0 ; j < SIZE ; j++ )
          pop[p][i][j] = ((random()&1048575) / 100000.0 - 5) * 2;
      
      /*  SELPOP to MUT1 will be severe mutations */
      for ( i = NEWPOP ; i < MUT1 ; i++ )
        {
          pop_copy( p, i, p, irand(NEWPOP) );
          dev = 1 + ((irand(2000) - 1000 )/ 5000);
          pop[p][i][irand(SIZE)] *= dev;
          dev = 1 + ((irand(2000) - 1000 )/ 5000);
          pop[p][i][irand(SIZE)] *= dev;
        }
  
      /* MUT2 to MAXPOP will be crossovers */
      for ( i = MUT1 ; i < MAXPOP ; i++ )
        {
          /* Every several generations (set by MIXGEN) there is a cross-over
             between different populations. */
          pop_copy( p, i, (((generation%MIXGEN)==0) ? irand(POPS) : p), irand(NEWPOP) );
          j = irand(NEWPOP);
          k1 = irand( SIZE - 1);
          k2 = irand( SIZE - 1 - k1 ) + k1 + 1;
          for ( m = k1 ; m <= k2 ; m++ ) pop[p][i][m] = pop[p][j][m];
          /* Mutate slightly */
          dev = 1 + ((irand(2000) - 1000 )/ 50000);
          pop[p][i][irand(SIZE)] *= dev;
        }
    }
  
  calc_score();
  sort_population();
  statistics( generation );
}


/*-----------------------------------------------------------------------*\
|                                                                         |
|  Return the number of cases for which the NN returns the correct value  |
|                                                                         |
\*-----------------------------------------------------------------------*/
check_performance()
{
  vector x;
  int j, count=0;
  for ( j = 0 ; j < total ; j++ )
    {
      x = test[j];
      if ( net(x) == hits[j] )
        count++;
    }
  return count;
}


/*-----------------------------------------------------------------------*\
|                                                                         |
|  Get data (read input file)                                             |
|                                                                         |
\*-----------------------------------------------------------------------*/
int get_data()
{
  char* FileName = "/tmp/nn-input";
  FILE *fd;
  int i, posnum, negnum;
  float x,y;

  /* opens the file  */
  if ( (fd = fopen(FileName,"r")) == NULL )
    {
      printf ("no-input-file");
      exit(10);
    }

  /* Total number of input values */
  total = 0;
  
  /* read the positive examples */
  fscanf( fd, "%d", &posnum);
  if (posnum > LIMIT)
    {
      printf("Error");
      exit(20);
    }
  for ( i = 0 ; i < posnum ; i++ )
    {
      fscanf( fd, "%f %f", &x, &y);
      test[ total ].p[0] = x / 1000;
      test[ total ].p[1] = y / 1000;
      hits[ total++ ] = 1;  /* 1 for positive examples */
    }

  /* read the negative examples */
  fscanf( fd, "%d", &negnum);
  if ((negnum+total) > LIMIT)
    {
      printf("Error");
      exit(21);
    }
  for ( i = 0 ; i < negnum ; i++ )
    {
      fscanf( fd, "%f %f", &x, &y);
      test[ total ].p[0] = x / 1000;
      test[ total ].p[1] = y / 1000;
      hits[ total++ ] = 0; /* 0 for negative example */
    }

  fclose( fd );
  return (0) ;
}


/*-----------------------------------------------------------------------*\
|                                                                         |
|   best_pop  -  Find the population with the best solution               |
|                                                                         |
\*-----------------------------------------------------------------------*/
int best_pop()
{
  int i, p, best = 0;

  for ( i = 0 ; i < POPS ; i++ )
    if ( score[i][0] > best )
      {
        best = score[i][0];
        p = i;
      }
  return(p);
}


/*-----------------------------------------------------------------------*\
|                                                                         |
|   charmap  -  draw a charmap showing the NN's behaviour                 |
|                                                                         |
\*-----------------------------------------------------------------------*/
charmap( p )
     int p;
{
  int i, j, result;
  vector x;

  apply( p ,0 );
  for ( i = 0 ; i < 350 ; i++ )
    {
      for ( j = 0 ; j < 350 ; j++ )
        if ( (i%12==0) && (j%6==0) )
          {
            x.p[0] = j/1000.0;
            x.p[1] = i/1000.0;
            result = net( x );
            printf("%c", (result==1 ? '+' : '.' ) );
          }
      if ( i%12==0 ) printf("\n");
    }
}


/*-----------------------------------------------------------------------*\
|                                                                         |
|   make_output  -  create the output file                                |
|                                                                         |
\*-----------------------------------------------------------------------*/
make_output(p)
     int p;
{
  int i, j, result, oldresult, start;
  vector x;
  char* FileName = "/tmp/nn-output";
  FILE *fd;
  
  printf("\n%s\n", (score[p][0]!=total ? "Failed." : "Success" ) );

  apply( p, 0 );
  
  printf("Writing output file...\n");
  /* Open the file  */
  if ( (fd = fopen(FileName,"w")) == NULL )
    {
      printf ("Can't open output file");
      exit(10);
    }

  /* line scheme */
  for ( i = 0 ; i < 350 ; i++ )  /* Scan horizontally */
    {
      result = 0;
      for ( j = 0 ; j < 350 ; j++ )
        {
          oldresult = result;
          x.p[0] = j/1000.0;
          x.p[1] = i/1000.0;
          result = net( x );
          if ( oldresult != result )
            fprintf( fd, "%d %d ", j, i );
        }
    }
  
  for ( j = 0 ; j < 350 ; j++ )  /* Scan vertically */
    {
      result = 0;
      for ( i = 0 ; i < 350 ; i++ )
        {
          oldresult = result;
          x.p[0] = j/1000.0;
          x.p[1] = i/1000.0;
          result = net( x );
          if ( oldresult != result )
            fprintf( fd, "%d %d ", j, i );
        }
    }
  fclose( fd );
  printf("Done!\n");
}


/*-----------------------------------------------------------------------*\
|                                                                         |
|  Main                                                                   |
|                                                                         |
\*-----------------------------------------------------------------------*/
main()
{
  int generation, j, p, best, done = 0;
  float px, py, px1, py1;

  randomize();
  get_data();  /* Read input from file */
  
  make_initial_population();
  calc_score();
  sort_population();

  /* Educate the net */
  generation = 0;
  while ( (done != 1 ) && ( generation++ < SESSIONS ) )
    {
      make_next_generation( generation );
      p = best_pop();
      /* Show a charmap every 50 generations */
      if ( generation % 50 == 0 ) charmap(p);
      if ( score[p][0] == total )
        done = 1;
    }

  /* return results */
  make_output(p);
}
