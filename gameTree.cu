#include <iostream>
#include <stdint.h>
#include <cstdio>
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
	bool valid;
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

const Board bad_board_host = {{empty}, false};
__constant__  Board bad_board = {{empty}, false};


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
__device__ int analyseBoard(Board *board)
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
	int base_board = tx/22;	
	int board_from_base = tx%22;

	unsigned int blockNum = blockIdx.x+blockIdx.y*gridDim.x;
	__shared__ int scores[512];
	__shared__ bool invalid[512];

	scores[tx] = input[blockNum*blockDim.x+tx];
	invalid[tx] = (scores[tx] > 20000);

	for(int stride = 2; stride <= 32; stride *= 2)
	{	
		if (board_from_base*(stride) + stride/2 < 22)
			scores[base_board+board_from_base*stride] = min(scores[base_board+board_from_base*stride], 
														scores[base_board+board_from_base*stride+stride/2]);
	}
	for( int stride = 2; stride <= 32; stride *= 2)
	{
		int index1 = base_board*stride*22;
		int index2 = base_board*stride*22+stride*11;
		if(base_board*stride+stride/2 < 22)
		{
			scores[base_board*stride*22] = max(invalid[index1]*-2000000+scores[index1],
											invalid[index2]*-2000000+scores[index2]);
		}
	}

	if (threadIdx.x == 0)
		output[blockNum] = scores[0];
	
	
}

//reduces by 1 turn, with boards at the leaf nodes
//works with 512 spawned threads
__global__ void analyze_board_tree(Board * input, int * output){
	int tx = threadIdx.x;
	int base_board = tx/22;	
	int board_from_base = tx%22;

	unsigned int blockNum = blockIdx.x+blockIdx.y*gridDim.x;
	__shared__ int scores[512];
	__shared__ bool invalid[512];

	scores[tx] = analyseBoard(&input[blockNum*blockDim.x+threadIdx.x]);
	invalid[tx] = (scores[tx] > 20000);

	for(int stride = 2; stride <= 32; stride *= 2)
	{	
		if (board_from_base*(stride) + stride/2 < 22)
			scores[base_board+board_from_base*stride] = min(scores[base_board+board_from_base*stride], 
														scores[base_board+board_from_base*stride+stride/2]);
		__syncthreads();
	}
	for( int stride = 2; stride <= 32; stride *= 2)
	{
		int index1 = base_board*stride*22;
		int index2 = base_board*stride*22+stride*11;
		if(base_board*stride+stride/2 < 22)
		{
			scores[base_board*stride*22] = max(invalid[index1]*-2000000+scores[index1],
											invalid[index2]*-2000000+scores[index2]);
		}
		__syncthreads();
	}

	if (threadIdx.x == 0)
		output[blockNum] = scores[0];
	
	
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
				if(!(y%2) && x != 0 && !b.pieces[x-1][y-1])
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

int main(int argc, char **argv) {
	Board * b = (Board *)malloc(sizeof(Board)*512);
	initBoard(b);
	makeMove(b);
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
	Board *device_input;

	int inputSize = 1;
	int outputSize = inputSize * 512;
	
	host_input =  board;

	if(USE_GPU)
	{
		// cuda malloc
		cudaMalloc(&device_output, outputSize * sizeof(Board));
		cudaMalloc(&device_input, inputSize * sizeof(Board));
		
		// cuda memcpy
		cudaMemcpy(device_input, host_input, inputSize * sizeof(*device_input), cudaMemcpyHostToDevice);

		//launch kernel and check errors
		//printf("initializing kernel with grid dim: %d and block dim: %d\n", inputSize, BLOCK_SIZE);
		dim3 dimGrid(inputSize);
		dim3 dimBlock(BLOCK_SIZE);
		expand<<<dimGrid, dimBlock>>>(device_input, device_output, inputSize);
		cudaPeekAtLastError();
		cudaDeviceSynchronize();

		//set up for second kernel launch
		inputSize = outputSize;
		outputSize = inputSize * 512;
		cudaFree(device_input);
		device_input = device_output;
		cudaMalloc(&device_output, outputSize * sizeof(Board));
		
		//launch kernel and check errors
		//printf("initializing kernel with grid dim: %d and block dim: %d\n", inputSize, BLOCK_SIZE);
		dim3 dimGrid2(inputSize);
		expand<<<dimGrid2, dimBlock>>>(device_input, device_output,	inputSize);
		cudaPeekAtLastError();
		cudaDeviceSynchronize();
		
		//print all boards after 2 full turns have been taken
		host_output = (Board *) malloc(outputSize*sizeof(*host_output));
		gpuErrChk(cudaMemcpy(host_output, device_output, outputSize * sizeof(*device_input),
					cudaMemcpyDeviceToHost));
		for(int i = 0; i < outputSize; i++)
		{
			if(false && !boardEquality(&bad_board_host, &host_output[i]))
			{
				printBoard(host_output[i]);		
				printf("Board #: %d", i);
			}
		}
		int expansion_rate = 32;
		dim3 dimGrid3(1*expansion_rate);
		dim3 dimGrid4(512*expansion_rate);
		
		Board *temp_device_output;
		Board *third_level_output;
		int * device_second_level_scores;
		int * device_third_level_scores;
		cudaMalloc(&device_second_level_scores, 512*512*sizeof(int));
		cudaMalloc(&device_third_level_scores, 512*expansion_rate*sizeof(int));
		gpuErrChk(cudaMalloc(&temp_device_output, 512*512*expansion_rate*sizeof(Board)));
		gpuErrChk(cudaMalloc(&third_level_output, 512*expansion_rate*sizeof(Board)));

		for(int i = 0; i < 512*512/expansion_rate; i++)
		{
			device_input = &device_output[i*expansion_rate];
			expand<<<dimGrid3, dimBlock>>>(device_input, third_level_output, expansion_rate);
			cudaPeekAtLastError();
			cudaDeviceSynchronize();
			
			expand<<<dimGrid2, dimBlock>>>(third_level_output, temp_device_output, 512*expansion_rate);
			cudaPeekAtLastError();
			cudaDeviceSynchronize();
			analyze_board_tree<<<dimGrid2, dimBlock>>>(temp_device_output, device_third_level_scores);
			analyze_score_tree<<<dimGrid4, dimBlock>>>(device_third_level_scores, 
														&device_second_level_scores[i*expansion_rate]);

		}
		int * second_level_scores = (int*)malloc(512*512*sizeof(int));
		Board * second_level_boards = (Board*)malloc(512*512*sizeof(Board));
		cudaMemcpy(second_level_scores, device_second_level_scores, 512*512*sizeof(int), cudaMemcpyDeviceToHost);
		cudaMemcpy(second_level_boards, device_output, 512*512*sizeof(Board), cudaMemcpyDeviceToHost);
		for(int i = 0; i < 512*512; i++)
		{	
			if(!boardEquality(&bad_board_host, &second_level_boards[i]))
			{
				printf("board %d, scores %d\n", i, second_level_scores[i]);
				printBoard(second_level_boards[i]);
			}
		}
		gpuErrChk(cudaFree(temp_device_output));
		gpuErrChk(cudaFree(device_output));
		gpuErrChk(cudaFree(device_third_level_scores));
	} else // iterative version
	{
		Turn turn = white;
		host_output = new Board[22*22*22*22];
		host_output[0] = *board;
		if(!host_output)
		{
			fprintf(stderr, "operator new failed on size %d\n", 22*22*22*22);
			return -1;
		}
		clock_t start = clock(), diff;
		makeMoves(host_output, turn, 0);
		
		turn = black;
		for(int i = 0; i < 22; i++)
		{
			if(boardIsValid_host(&host_output[i]))
			{
				makeMoves(host_output, turn, i);
			}
		}

		//printBoard(host_output[0]);
		turn = white;
		for(int i = 0; i < 22 * 22; i++)
		{
			if(boardIsValid_host(&host_output[i]))
			{
				makeMoves(host_output, turn, i);
			}
		}
		
		//printBoard(host_output[0]);
		turn = black;
		for(int i = 0; i < 22 * 22 * 22; i++)
		{
			if(boardIsValid_host(&host_output[i]))
			{
				makeMoves(host_output, turn, i);
			}
		}
		diff = clock() - start;
		int msec = diff * 1000 / CLOCKS_PER_SEC;
		printf("Time taken %d seconds %d milliseconds\n", msec/1000, msec%1000);
		//printBoard(host_output[0]);
		
		int sum = 0, last_idx;
		for(int i = 0; i < 22 * 22 * 22 * 22; i++)
		{
			if(boardIsValid_host(&host_output[i]))
			{
				sum++;
				last_idx = i;
			}
		}
		
		printf("%d %d\n", sum, last_idx);
		printBoard(host_output[last_idx]);
		
		*board = host_output[0];
	}
	return 0;
}

int analyseBoard(Board *board, Turn player)
{
	int score = 0;
	uint8_t pieceMin, pieceMax;
	if(player == white)
	{
		pieceMin = white_reg;
		pieceMax = white_king_moved;
	} else
	{
		pieceMin = black_reg;
		pieceMax = black_reg_moved;
	}

	for(int x = 0; x < 4; x++)
	{
		for(int y = 0; y < 8; y++)
		{
			Piece piece = board->pieces[x][y];
			if(pieceMin <= piece && piece <= pieceMax)
			{
				score++;
			}
		}
	}
	return score;		
}
