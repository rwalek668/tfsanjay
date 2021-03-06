#include <algorithm>
#include <math.h>
#include <iostream>
#include <stdint.h>
#include <stdio.h>
#include <time.h>

#define USE_GPU 1

/*
enum Piece
{
	empty,
	white_reg,
	white_reg_moved,
	white_king,
	white_king_moved,
	black_reg,
	black_reg_moved,
	black_king,
	black_king_moved
};*/

typedef uint8_t Piece;
const Piece empty = 0;
const Piece white_reg = empty + 1;
const Piece white_reg_moved = white_reg + 1;
const Piece white_king = white_reg_moved + 1;
const Piece white_king_moved = white_king + 1;
const Piece black_reg = white_king_moved + 1;
const Piece black_reg_moved = black_reg + 1;
const Piece black_king = black_reg_moved + 1;
const Piece black_king_moved = black_king + 1;

struct Board {
	Piece pieces[4][8];
	//bool valid;
};
enum Turn
{
	white,
	black
};
struct Pair {
	unsigned char first;
	unsigned char second;
};

const Board bad_board_host = {{empty}};//, false};
__constant__  Board bad_board = {{empty}};//, false};


#define BLOCK_SIZE 512
#define gpuErrChk(stmt) \
do\
{\
	cudaError_t errCode = stmt; \
	if(errCode != cudaSuccess)\
	{ \
		std::cerr << "gpuErrChk: " << cudaGetErrorString(errCode)\
			<< " " <<  __FILE__ << " " <<  __LINE__ << " "\
			<< std::endl;\
		return -1;\
	}\
} while(0)

__device__ Board outputBoard;
__host__ __device__ void makeMoves(Board * boards, Turn turn, unsigned int tx);

__host__ __device__ int ipow(int base, int exp)
{
	int result = 1;
	while(exp)
	{
		if(exp & 1)
		{
			result *= base;
		}
		exp >>= 1;
		base *= base;
	}
	return result;
}

__host__ __device__ bool boardEquality(const Board *a, const Board *b)
{
	for(int x = 0; x < 4; x++)
	{
		for(int y = 0; y < 8; y++)
		{
			if(a->pieces[x][y] != b->pieces[x][y])
			{
				return false;
			}
		}
	}
	return true;
}

__host__ bool boardIsValid_host(const Board *a)
{
	return !boardEquality(a, &bad_board_host);
}

__device__ bool boardIsValid_device(const Board *a)
{
	return !boardEquality(a, &bad_board);
}
__host__ __device__ int analyseBoard(Board *board)
{
	int score = 0;
	int white_wins = 1;
	int black_wins = 1;
	for(int x = 0; x < 4; x++)
	{
		for(int y = 0; y < 8; y++)
		{
			//kings are worth 2, pieces are worth 1
			Piece piece = board->pieces[x][y];
			if (piece != empty && piece <= white_king_moved)
			{
				score += (piece+1)/2;
				white_wins = 0;		
			}
			else if (piece != empty)
			{
				score -= (piece-3)/2;
				black_wins = 0;
			}
		}
	}
	score = score + white_wins*10000 + black_wins*-10000;	
	//returns 1,000,000 if invalid board, 
	return score*(!(white_wins && black_wins)) + 1000000*(white_wins && black_wins);	
}


//reduces by 1 turn, with scores at the leaf nodes
//works with 512 spawned threads
__global__ void analyze_score_tree(int * input, int * output){
	int tx = threadIdx.x;

	unsigned int blockNum = blockIdx.x+blockIdx.y*gridDim.x;
	__shared__ int scores[512];
	__shared__ int mins[22];

	scores[tx] = input[blockNum*blockDim.x+tx];
	__syncthreads();

	if(threadIdx.x < 22)
	{
		int min = 1000000;
		for(int i = 0; i < 22; i++)
		{
			int temp = scores[threadIdx.x*22+i];
			if (temp < min && temp != -100000000)
				min = temp;
		}
		mins[threadIdx.x] = min;
	}
	__syncthreads();
	if(threadIdx.x == 0)
	{
		int max = -100000000;
		for(int i = 0; i < 22; i++)
			if(mins[i] > max && mins[i] != 1000000)
				max = mins[i];
		output[blockNum] = max;
	}
	
	
}

//reduces by 1 turn, with boards at the leaf nodes
//works with 512 spawned threads
__global__ void analyze_board_tree(Board * input, int * output){
	int tx = threadIdx.x;

	unsigned int blockNum = blockIdx.x+blockIdx.y*gridDim.x;
	__shared__ int scores[512];
	__shared__ int mins[22];

	scores[tx] = analyseBoard(&input[blockNum*blockDim.x+threadIdx.x]);
	__syncthreads();

	if(threadIdx.x < 22)
	{
		int min = 1000000;
		for(int i = 0; i < 22; i++)
		{
			int temp = scores[threadIdx.x*22+i];
			if (temp < min)
				min = temp;
		}
		mins[threadIdx.x] = min;
	}
	__syncthreads();
	if(threadIdx.x == 0)
	{
		int max = -100000000;
		for(int i = 0; i < 22; i++)
			if(mins[i] > max && mins[i] != 1000000)
				max = mins[i];
		output[blockNum] = max;
	}
/*
	for(int stride = 2; stride <= 32; stride *= 2)
	{	
		if (board_from_base*(stride) + stride/2 < 22 && board_from_base%stride == 0)
			if(scores[base_board+board_from_base*stride+stride/2] < scores[base_board+board_from_base*stride])
			scores[base_board+board_from_base*stride] = scores[base_board+board_from_base*stride+stride/2];
		__syncthreads();
	}
	for( int stride = 2; stride <= 32; stride *= 2)
	{
		int index1 = base_board*stride*22;
		int index2 = base_board*stride*22+stride*11;
		if(base_board*stride+stride/2 < 22 && base_board%stride == 0)
		{
			if( scores[index1] < scores[index2] && scores[index2] != 1000000)
				scores[base_board*stride*22] = scores[index2];
			if (scores[base_board*stride*22] == 1000000)
				scores[base_board*stride*22] = -1000000;
		}
		__syncthreads();
	}

	if (threadIdx.x == 0)
		output[blockNum] = scores[0];*/
}

__global__ void expand(Board * input, Board * output, int len) {
	const int shared_size = 484;
	__shared__ Board B[shared_size]; //TODO
	unsigned int tx = threadIdx.x;
	unsigned int blockNum = blockIdx.x+blockIdx.y*gridDim.x;
	
	if (blockNum < len && tx == 0)
		B[0] = input[blockNum];
	else if (blockNum < len && tx < shared_size)
		B[tx] = bad_board;

	__syncthreads();
	if(tx == 0 && ~boardEquality(&B[tx], &bad_board))
		makeMoves(B, white, tx);
	__syncthreads();
	if(tx < shared_size && ~boardEquality(&B[tx], &bad_board))
		makeMoves(B, black, tx);
	__syncthreads();

	if (tx < shared_size && blockNum < len)
		output[blockDim.x*blockNum+tx] = B[tx];
	else if (blockNum < len)
		output[blockDim.x*blockNum+tx] = bad_board;
}

//TODO: deal with 22 move boundary
__host__ __device__ 
void makeMoves(Board * boards, Turn turn, unsigned int tx)
{
	// tx = 0 condition because only the first thread has a valid board to work on.
	if(turn == white && tx == 0)
	{
		int exp_rate = 22;
		int move_idx = 0;
		Board b = boards[tx];
		Board temp = boards[tx];
		for(int x = 0; x < 4; x++)
		for(int y = 0; y < 8; y++)
		{
			if (b.pieces[x][y] == white_reg || b.pieces[x][y] == white_king)
			{
				/*White pieces move (not take) */
				if(y%2 && y < 6 && x != 3 && !b.pieces[x+1][y+1]) 
				{	
					//printf("white at %d,%d move right\n", x, y);
					temp.pieces[x+1][y+1] = temp.pieces[x][y];
					temp.pieces[x][y] = empty;
					boards[tx+move_idx*exp_rate] = temp;
					move_idx++;
					temp = b;
				}
				if(y%2 && y < 6 && !b.pieces[x][y+1])
				{
					//printf("white at %d,%d move left\n", x, y);
					temp.pieces[x][y+1] = temp.pieces[x][y];
					temp.pieces[x][y] = empty;
					boards[tx+move_idx*exp_rate] = temp;
					move_idx++;
					temp = b;
				}
				if(!(y%2) && x != 0 && !b.pieces[x-1][y+1])
				{
					//printf("white at %d,%d move left\n", x, y);
					if (y == 6)
						temp.pieces[x-1][y+1] = white_king;
					else
						temp.pieces[x-1][y+1] = temp.pieces[x][y];
					temp.pieces[x][y] = empty;
					boards[tx+move_idx*exp_rate] = temp;
					move_idx++;
					temp = b;
				}
				if(!(y%2) && !b.pieces[x][y+1])
				{
					//printf("white at %d,%d move right\n", x, y);
					if (y == 6)
						temp.pieces[x][y+1] = white_king;
					else
						temp.pieces[x][y+1] = temp.pieces[x][y];
					temp.pieces[x][y] = empty;
					boards[tx+move_idx*exp_rate] = temp;
					move_idx++;
					temp = b;
				}
				/*White piece captures a black piece (not become king)*/
				if(y%2 && x!= 3 && b.pieces[x+1][y+1] > white_king_moved && !b.pieces[x+1][y+2]) 
				{
					//TODO add double takes here
					if (y != 5)
						temp.pieces[x+1][y+2] = temp.pieces[x][y];
					else
						temp.pieces[x+1][y+2] = white_king;
					temp.pieces[x][y] = empty;
					temp.pieces[x+1][y+1] = empty;
					boards[tx+move_idx*exp_rate] = temp;
					move_idx++;
					temp = b;
				}
				if(y%2 && x != 0 && b.pieces[x][y+1] > white_king_moved && !b.pieces[x-1][y+2])
				{
					//TODO add double takes here
					if (y != 5)
						temp.pieces[x-1][y+2] = temp.pieces[x][y];
					else
						temp.pieces[x+1][y+2] = white_king;
					temp.pieces[x][y] = empty;
					temp.pieces[x][y+1] = empty;
					boards[tx+move_idx*exp_rate] = temp;
					move_idx++;
					temp = b;
				}
				if(!(y%2) && y < 5 && x != 0 && b.pieces[x-1][y+1] > white_king_moved && !b.pieces[x-1][y+2])
				{
					//TODO add double takes here
					temp.pieces[x-1][y+2] = temp.pieces[x][y];
					temp.pieces[x][y] = empty;
					temp.pieces[x-1][y+1] = empty;
					boards[tx+move_idx*exp_rate] = temp;
					move_idx++;
					temp = b;
				}
				if(!(y%2) && y < 5 && x != 3 && b.pieces[x][y+1] > white_king_moved && !b.pieces[x+1][y+2])
				{
					//TODO add double takes here
					temp.pieces[x+1][y+2] = temp.pieces[x][y];
					temp.pieces[x][y] = empty;
					temp.pieces[x][y+1] = empty;
					boards[tx+move_idx*exp_rate] = temp;
					move_idx++;
					temp = b;
				}
			}
			if (b.pieces[x][y] == white_king)
			{
		
				/*White king move backwards(not take) */
				if(y%2 && x != 3 && !b.pieces[x+1][y-1]) 
				{
					temp.pieces[x+1][y-1] = temp.pieces[x][y];
					temp.pieces[x][y] = empty;
					boards[tx+move_idx*exp_rate] = temp;
					move_idx++;
					temp = b;
				}
				if(y%2 && !b.pieces[x][y-1])
				{
					temp.pieces[x][y-1] = temp.pieces[x][y];
					temp.pieces[x][y] = empty;
					boards[tx+move_idx*exp_rate] = temp;
					move_idx++;
					temp = b;
				}
				if(!(y%2) && y>0 && x != 0 && !b.pieces[x-1][y-1])
				{
					temp.pieces[x-1][y-1] = temp.pieces[x][y];
					temp.pieces[x][y] = empty;
					boards[tx+move_idx*exp_rate] = temp;
					move_idx++;
					temp = b;
				}
				if(!(y%2) && y>0 && !b.pieces[x][y-1])
				{
					temp.pieces[x][y-1] = temp.pieces[x][y];
					temp.pieces[x][y] = empty;
					boards[tx+move_idx*exp_rate] = temp;
					move_idx++;
					temp = b;
				}
				if(y%2 && y>1 && x!= 3 && b.pieces[x+1][y-1] > white_king_moved && !b.pieces[x+1][y-2]) 
				{
					//TODO add double takes here
					temp.pieces[x+1][y-2] = temp.pieces[x][y];
					temp.pieces[x][y] = empty;
					temp.pieces[x+1][y-1] = empty;
					boards[tx+move_idx*exp_rate] = temp;
					move_idx++;
					temp = b;
				}
				if(y%2 && y>1 && x != 0 && b.pieces[x][y-1] > white_king_moved && !b.pieces[x-1][y-2])
				{
					//TODO add double takes here
					temp.pieces[x-1][y-2] = temp.pieces[x][y];
					temp.pieces[x][y] = empty;
					temp.pieces[x][y-1] = empty;
					boards[tx+move_idx*exp_rate] = temp;
					move_idx++;
					temp = b;
				}
				if(!(y%2) && y>0 && x != 0 && b.pieces[x-1][y-1] > white_king_moved && !b.pieces[x-1][y-2])
				{
					//TODO add double takes here
					temp.pieces[x-1][y-2] = temp.pieces[x][y];
					temp.pieces[x][y] = empty;
					temp.pieces[x-1][y-1] = empty;
					boards[tx+move_idx*exp_rate] = temp;
					move_idx++;
					temp = b;
				}
				if(!(y%2) && y>0 && x!=3 && b.pieces[x][y-1] > white_king_moved && !b.pieces[x+1][y-2])
				{
					//TODO add double takes here
					temp.pieces[x+1][y-2] = temp.pieces[x][y];
					temp.pieces[x][y] = empty;
					temp.pieces[x][y-1] = empty;
					boards[tx+move_idx*exp_rate] = temp;
					move_idx++;
					temp = b;
				}
			}
		}
	}
	else if (tx < 22)
	{
		int move_idx = 0;
		Board b = boards[tx*22];
		Board temp = boards[tx*22];
		for(int x = 0; x < 4; x++)
		for(int y = 0; y < 8; y++)
		{
			if (b.pieces[x][y] == black_reg || b.pieces[x][y] == black_king)
			{
				/*White pieces move (not take) */
				if(y%2 && x != 3 && !b.pieces[x+1][y-1]) 
				{	
					//printf("black at %d,%d move right\n", x, y);
					if (y == 1)
						temp.pieces[x+1][y-1] = black_king;
					else
						temp.pieces[x+1][y-1] = temp.pieces[x][y];
					temp.pieces[x][y] = empty;
					boards[22*tx+move_idx] = temp;
					move_idx++;
					temp = b;
				}
				if(y%2 && !b.pieces[x][y-1])
				{
					//printf("black at %d,%d move left\n", x, y);
					if (y == 1)
						temp.pieces[x+1][y-1] = black_king;
					else
						temp.pieces[x][y-1] = temp.pieces[x][y];
					temp.pieces[x][y] = empty;
					boards[22*tx+move_idx] = temp;
					move_idx++;
					temp = b;
				}
				if(!(y%2) && y != 0 && x != 0 && !b.pieces[x-1][y-1])
				{
					//printf("black at %d,%d move left\n", x, y);
					temp.pieces[x-1][y-1] = temp.pieces[x][y];
					temp.pieces[x][y] = empty;
					boards[22*tx+move_idx] = temp;
					move_idx++;
					temp = b;
				}
				if(!(y%2) && !b.pieces[x][y-1])
				{
					//printf("black at %d,%d move right\n", x, y);
					temp.pieces[x][y-1] = temp.pieces[x][y];
					temp.pieces[x][y] = empty;
					boards[22*tx+move_idx] = temp;
					move_idx++;
					temp = b;
				}
				/*White piece captures a black piece*/
				if(y%2 && y>1 && x!= 3 && b.pieces[x+1][y-1] > 0 && b.pieces[x+1][y-1] <= white_king_moved && !b.pieces[x+1][y-2]) 
				{
					//TODO add double takes here
					if (y != 2)
						temp.pieces[x+1][y-2] = temp.pieces[x][y];
					else
						temp.pieces[x+1][y-2] = white_king;
					temp.pieces[x][y] = empty;
					temp.pieces[x+1][y-1] = empty;
					boards[22*tx+move_idx] = temp;
					move_idx++;
					temp = b;
				}
				if(y%2 && y>1 && x != 0 && b.pieces[x][y-1] > 0 && b.pieces[x][y-1] <= white_king_moved && !b.pieces[x-1][y-2])
				{
					//TODO add double takes here
					if (y != 2)
						temp.pieces[x-1][y-2] = temp.pieces[x][y];
					else
						temp.pieces[x+1][y-2] = white_king;
					temp.pieces[x][y] = empty;
					temp.pieces[x][y-1] = empty;
					boards[22*tx+move_idx] = temp;
					move_idx++;
					temp = b;
				}
				if(!(y%2) && y>2 && x != 0 && b.pieces[x-1][y-1] <= white_king_moved && b.pieces[x-1][y-1] > 0 && !b.pieces[x-1][y-2])
				{
					//TODO add double takes here
					temp.pieces[x-1][y-2] = temp.pieces[x][y];
					temp.pieces[x][y] = empty;
					temp.pieces[x-1][y-1] = empty;
					boards[22*tx+move_idx] = temp;
					move_idx++;
					temp = b;
				}
				if(!(y%2) && y>2 && x!=3 && b.pieces[x][y-1] <= white_king_moved && b.pieces[x][y-1]>0 && !b.pieces[x+1][y-2])
				{
					//TODO add double takes here
					temp.pieces[x+1][y-2] = temp.pieces[x][y];
					temp.pieces[x][y] = empty;
					temp.pieces[x][y-1] = empty;
					boards[22*tx+move_idx] = temp;
					move_idx++;
					temp = b;
				}
			}
			if (b.pieces[x][y] == black_king)
			{
		
				/*White king move backwards(not take) */
				if(y%2 && y<7 && x != 3 && !b.pieces[x+1][y+1]) 
				{
					temp.pieces[x+1][y+1] = temp.pieces[x][y];
					temp.pieces[x][y] = empty;
					boards[22*tx+move_idx] = temp;
					move_idx++;
					temp = b;
				}
				if(y%2 && y<7 && !b.pieces[x][y+1])
				{
					temp.pieces[x][y+1] = temp.pieces[x][y];
					temp.pieces[x][y] = empty;
					boards[22*tx+move_idx] = temp;
					move_idx++;
					temp = b;
				}
				if(!(y%2) && x != 0 && !b.pieces[x-1][y+1])
				{
					temp.pieces[x-1][y+1] = temp.pieces[x][y];
					temp.pieces[x][y] = empty;
					boards[22*tx+move_idx] = temp;
					move_idx++;
					temp = b;
				}
				if(!(y%2) && !b.pieces[x][y+1])
				{
					temp.pieces[x][y+1] = temp.pieces[x][y];
					temp.pieces[x][y] = empty;
					boards[22*tx+move_idx] = temp;
					move_idx++;
					temp = b;
				}
				if(y%2 && y<6 && x!= 3 && b.pieces[x+1][y+1] <= white_king_moved && b.pieces[x+1][y+1] > 0 && !b.pieces[x+1][y+2]) 
				{
					//TODO add double takes here
					temp.pieces[x+1][y+2] = temp.pieces[x][y];
					temp.pieces[x][y] = empty;
					temp.pieces[x+1][y+1] = empty;
					boards[22*tx+move_idx] = temp;
					move_idx++;
					temp = b;
				}
				if(y%2 && y<6 && x != 0 && b.pieces[x][y+1] <= white_king_moved && b.pieces[x][y+1] > 0 && !b.pieces[x-1][y+2])
				{
					//TODO add double takes here
					temp.pieces[x-1][y+2] = temp.pieces[x][y];
					temp.pieces[x][y] = empty;
					temp.pieces[x][y+1] = empty;
					boards[22*tx+move_idx] = temp;
					move_idx++;
					temp = b;
				}
				if(!(y%2) && y<5 && x != 0 && b.pieces[x-1][y+1] <= white_king_moved && b.pieces[x-1][y+1] > 0 && !b.pieces[x-1][y+2])
				{
					//TODO add double takes here
					temp.pieces[x-1][y+2] = temp.pieces[x][y];
					temp.pieces[x][y] = empty;
					temp.pieces[x-1][y+1] = empty;
					boards[22*tx+move_idx] = temp;
					move_idx++;
					temp = b;
				}
				if(!(y%2) && y<5 && x!=3 && b.pieces[x][y+1] <= white_king_moved && b.pieces[x][y+1] > 0 && !b.pieces[x+1][y+2])
				{
					//TODO add double takes here
					temp.pieces[x+1][y+2] = temp.pieces[x][y];
					temp.pieces[x][y] = empty;
					temp.pieces[x][y+1] = empty;
					boards[22*tx+move_idx] = temp;
					move_idx++;
					temp = b;
				}
			}
		}
	}
} 

void printBoard(Board b);
int initBoard(Board *b);
int makeMove(Board *board);
int analyseBoard(Board *board, Turn player);
void reverse(Board * b);

int main(int argc, char **argv) {
	Board * b = (Board *)malloc(sizeof(Board)*512);
	initBoard(b);
	for(int i = 0; i <100; i++)
	{
		clock_t start = clock(), diff;
		makeMove(b);
		if(i%2)
		printBoard(b[0]);
		reverse(b);
		//printBoard(b[0]);
		diff = clock() - start;
		int msec = diff * 1000 / CLOCKS_PER_SEC;
		//printf("Time taken %d seconds %d milliseconds\n", msec/1000, msec%1000);
	}
	
}
void reverse(Board * b)
{
	Piece temp;
	for(int i = 0; i < 2; i++)
		for(int j = 0; j < 8; j++)
		{	
			temp = b->pieces[i][j];
			if(b->pieces[3-i][7-j] > 4)
				b->pieces[i][j] = b->pieces[3-i][7-j]-4;
			else if(b->pieces[3-i][7-j] <= 4 && b->pieces[3-i][7-j] > 0)
				b->pieces[i][j] = b->pieces[3-i][7-j]+4;
			else
				b->pieces[i][j] = b->pieces[3-i][7-j];
			if(temp > 4)
				b->pieces[3-i][7-j] = temp-4;
			else if(temp <= 4 && temp > 0)
				b->pieces[3-i][7-j] = temp+4;
			else
				b->pieces[3-i][7-j] = temp;
		}
}
void printBoard(Board b)
{
	printf("Board: --------------------------------------\n");
	for(int i = 3; i >= 0; i--)
	{
		for(int j = 0; j < 4; j++)
		{
			switch(b.pieces[j][i*2+1])
			{
				case white_reg:
				case white_reg_moved:
					printf("_|w|");
					break;
				case white_king:
				case white_king_moved:
					printf("_|W|");
					break;
				case black_reg:
				case black_reg_moved:
					printf("_|b|");
					break;
				case black_king:
				case black_king_moved:
					printf("_|B|");
					break;
				case empty:
					printf("_|_|");
					break;
				default:
					printf("x|x|");
					break;
			}
		}
		printf("\n");
		for(int j = 0; j < 4; j++)
		{
			switch(b.pieces[j][i*2])
			{
				case white_reg:
				case white_reg_moved:
					printf("w|_|");
					break;
				case white_king:
				case white_king_moved:
					printf("W|_|");
					break;
				case black_reg:
				case black_reg_moved:
					printf("b|_|");
					break;
				case black_king:
				case black_king_moved:
					printf("B|_|");
					break;
				case empty:
					printf("_|_|");
					break;
				default:
					printf("x|x|");
					break;
			}
		}
		printf("\n");
	}
}

int initBoard(Board *board)
{
	if(!board)
	{
		return -1;
	}
	for(int y = 0; y < 3; y++)
	{
		for(int x = 0; x < 4; x++)
		{
			board->pieces[x][y] = white_reg;
			board->pieces[x][y + 5] = black_reg;
		}
	}
	return 0;
}

int makeMove(Board *board)
{
	Board *host_output;
	Board *host_input;
	Board *device_output;
	Board *device1_output;
	Board *device2_output;
	Board *device_input;

	int inputSize = 1;
	int outputSize = inputSize * 512;
	
	host_input =  board;
	if(USE_GPU)
	{
		// cuda malloc
		cudaMalloc(&device_input, inputSize * sizeof(Board));
		cudaMalloc(&device1_output, outputSize * sizeof(Board));
		cudaMalloc(&device2_output, outputSize * 512 * sizeof(Board));
		
		// cuda memcpy
		cudaMemcpy(device_input, host_input, inputSize * sizeof(*device_input), cudaMemcpyHostToDevice);

		//launch kernel and check errors
		//printf("initializing kernel with grid dim: %d and block dim: %d\n", inputSize, BLOCK_SIZE);
		dim3 dimGrid(1);
		dim3 dimBlock(BLOCK_SIZE);
		expand<<<dimGrid, dimBlock>>>(device_input, device1_output, inputSize);
		cudaPeekAtLastError();
		cudaDeviceSynchronize();

		//set up for second kernel launch
		inputSize = outputSize;
		outputSize = inputSize * 512;
		
		//launch kernel and check errors
		//printf("initializing kernel with grid dim: %d and block dim: %d\n", inputSize, BLOCK_SIZE);
		dim3 dimGrid2(512);
		expand<<<dimGrid2, dimBlock>>>(device1_output, device2_output,	inputSize);
		cudaPeekAtLastError();
		cudaDeviceSynchronize();
		
		int expansion_rate = 512;
		dim3 dimGrid3(1*expansion_rate);
		dim3 dimGrid4(512*expansion_rate);
		
		//Board *temp_device_output;
		Board *third_level_output;
		int * device_first_level_scores;
		int * device_second_level_scores;
		int * device_third_level_scores;
		cudaMalloc(&device_second_level_scores, 512*512*sizeof(int));
		cudaMalloc(&device_third_level_scores, 512*expansion_rate*sizeof(int));
		cudaMalloc(&device_first_level_scores, 512*sizeof(int));
		//gpuErrChk(cudaMalloc(&temp_device_output, 512*512*expansion_rate*sizeof(Board)));
		gpuErrChk(cudaMalloc(&third_level_output, 512*expansion_rate*sizeof(Board)));

		for(int i = 0; i < 512*512/expansion_rate; i++)
		{
			device_input = &device2_output[i*expansion_rate];
			expand<<<dimGrid3, dimBlock>>>(device_input, third_level_output, expansion_rate);
			cudaPeekAtLastError();
			cudaDeviceSynchronize();
			
			//expand<<<dimGrid4, dimBlock>>>(third_level_output, temp_device_output, 512*expansion_rate);
			//cudaPeekAtLastError();
			//cudaDeviceSynchronize();
			//analyze_board_tree<<<dimGrid4, dimBlock>>>(temp_device_output, device_third_level_scores);
			//cudaPeekAtLastError();
			//cudaDeviceSynchronize();
			analyze_board_tree<<<dimGrid3, dimBlock>>>(third_level_output, 
														&device_second_level_scores[i*expansion_rate]);
			cudaPeekAtLastError();
			cudaDeviceSynchronize();
		}
		analyze_score_tree<<<dimGrid2, dimBlock>>>(device_second_level_scores, 
													device_first_level_scores);
		cudaPeekAtLastError();
		cudaDeviceSynchronize();
		int * first_level_scores = (int*)malloc(512*sizeof(int));
		Board * second_level_boards = (Board*)malloc(512*512*sizeof(Board));
		cudaMemcpy(first_level_scores, device_first_level_scores, 512*sizeof(int), cudaMemcpyDeviceToHost);
		int max = -100000;
		int index = -1;
		for(int i = 0; i < 22; i++)
		{	
			int min = 1000000;	
			for(int j = 0; j < 22; j++)
				if(first_level_scores[22*i+j] < min && first_level_scores[22*i+j] != -100000000)
					min = first_level_scores[22*i+j];
			if (min > max && min != 1000000)
			{
				index = i;
				max = min;
			}
		}
		Board boards[512];
		boards[0] = host_input[0];
		makeMoves(boards, white, 0);
		host_input[0] = boards[22*index];

		cudaFree(device_second_level_scores);
		cudaFree(device_third_level_scores);
		cudaFree(device_first_level_scores);
		cudaFree(third_level_output);
		cudaFree(device_input);
		cudaFree(device1_output);
		cudaFree(device2_output);
		free(first_level_scores);
		free(second_level_boards);
		return 0;
	} else // iterative version
	{
		static int numTurns = 0;
		int score = 0;
		unsigned long size;
		if(!numTurns)
		{
			std::cin >> numTurns;
		}

		
		if(numTurns == 4)
		{
			size = ipow(512, 3);
		} else if(numTurns <= 3)
		{
			size = ipow(512, numTurns);
		} else
		{
			printf("max 4\n");
			return -1;
		}

		host_output = new (std::nothrow) Board[size];
		if(!host_output)
		{
			fprintf(stderr, "operator new failed on size %lu\n", size);
			return -1;
		}
		host_output[0] = *board;

		for(int i = 0; i < numTurns && i < 3; i++)
		{
			Board *temp_output = new (std::nothrow) Board[size];
			if(!temp_output)
			{
				fprintf(stderr, "new failed on size %lu\n", size);
				return -1;
			}

			for( int j = 0; j < ipow(512, i); j++)
			{
				if(!boardIsValid_host(&host_output[j]))
				{
					continue;
				}
				Board b[512] = {empty};
				b[0] = host_output[j];
				makeMoves(b, white, 0);
				for(int k = 0; k < 512; k++)
				{
					if(boardIsValid_host(&b[k]))
					{
						makeMoves(b, black, k);
					}
					temp_output[512 * j + k] = b[k];
				}
			}
			delete[] host_output;
			host_output = temp_output;
		}

		if(numTurns > 3)
		{
			for(int i = 0; i < ipow(512,3); i++)
			{
				Board b[512] = {empty};
				//Board *temp_output = new (std::nothrow) Board[ipow(512,2)];
				b[0] = host_output[i];
				makeMoves(b, white, 0);
				for(int j = 0; j < 512; j+=22)
				{
					if(boardIsValid_host(&host_output[i]))
					{
						makeMoves(b, black, j);
					}
				}
				for(int j = 0; j < 512; j++)
				{
					if(boardIsValid_host(&host_output[i]))
					{
						score = std::max(score, analyseBoard(&host_output[i]));
					}
				}
				//delete[] temp_output;
			}
		} else
		{
			int * scores = new int[ipow(512,numTurns - 1)];
			int max = 0, idx = -1;
			for(int i = numTurns; i > 0; i--)
			{
				for(int j = 0; j < ipow(512,i); j++)
				{
					if(boardIsValid_host(&host_output[j]))
					{
						score = std::max(score, analyseBoard(&host_output[i]));
					}
					if(!(j % 512))
					{
						scores[j/512] = score;
						if(score > max)
						{
							max = score;
							idx = j/512;
						}
						score = 0;
					}
				}
			}
			//printf("%d, %d\n", max, idx);
			Board boards[512];
			boards[0] = board[0];
			makeMoves(boards, white, 0);
			board[0] = boards[0];
		}
		//printf("Score: %d\n", score);
		delete [] host_output;
		/*	
		int sum = 0, last_idx;
		for(int i = 0; i < size; i++)
		{
			if(boardIsValid_host(&host_output[i]))
			{
				sum++;
				last_idx = i;
				//printBoard(host_output[i]);
			}
		}
		
		printf("%d %d\n", sum, last_idx);
		printBoard(host_output[last_idx]);
		*/
		*board = host_output[0];
	}
	return 0;
}


