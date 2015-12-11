#include <iostream>
#include <stdint.h>


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

__constant__ Board bad_board = {empty};

#define USE_GPU 1
#if USE_GPU

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
__device__ void makeMoves(Board * boards, Turn turn, unsigned int tx);

__device__ bool boardEquality(const Board *a, const Board*b)
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
__global__ void analyze_tree(Board * input, int moves){
	int max = 0;
}

__global__ void expand(Board * input, Board * output, int len) {
	__shared__ Board B[500]; //TODO
	unsigned int tx = threadIdx.x;
	unsigned int blockNum = blockIdx.x+blockIdx.y*gridDim.x;
	
	if (blockNum < len && tx == 0)
	{
		B[0] = input[blockNum];
	}
	else if (blockNum < len)
	{
		B[tx] = bad_board;
	}	
	__syncthreads();
	if(boardEquality(&B[tx], &bad_board))
		makeMoves(B, white, tx);
	__syncthreads();
	if(boardEquality(&B[tx], &bad_board))
		makeMoves(B, black, tx);
	__syncthreads();

}


//TODO: deal with 22 move boundary
__device__ 
#endif
void makeMoves(Board * boards, Turn turn, unsigned int tx)
{

	if(turn == white)
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
				if(!(y%2) && y < 5 && b.pieces[x][y+1] > white_king_moved && !b.pieces[x+1][y+2])
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
				if(!(y%2) && y>0 && b.pieces[x][y-1] > white_king_moved && !b.pieces[x+1][y-2])
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
	else
	{
		int move_idx = 0;
		Board b = boards[tx];
		Board temp = boards[tx];
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
					boards[tx+move_idx] = temp;
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
					boards[tx+move_idx] = temp;
					move_idx++;
					temp = b;
				}
				if(!(y%2) && x != 0 && !b.pieces[x-1][y-1])
				{
					//printf("black at %d,%d move left\n", x, y);
					temp.pieces[x-1][y-1] = temp.pieces[x][y];
					temp.pieces[x][y] = empty;
					boards[tx+move_idx] = temp;
					move_idx++;
					temp = b;
				}
				if(!(y%2) && !b.pieces[x][y-1])
				{
					//printf("black at %d,%d move right\n", x, y);
					temp.pieces[x][y-1] = temp.pieces[x][y];
					temp.pieces[x][y] = empty;
					boards[tx+move_idx] = temp;
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
					boards[tx+move_idx] = temp;
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
					boards[tx+move_idx] = temp;
					move_idx++;
					temp = b;
				}
				if(!(y%2) && y>2 && x != 0 && b.pieces[x-1][y-1] <= white_king_moved && b.pieces[x-1][y-1] > 0 && !b.pieces[x-1][y-2])
				{
					//TODO add double takes here
					temp.pieces[x-1][y-2] = temp.pieces[x][y];
					temp.pieces[x][y] = empty;
					temp.pieces[x-1][y-1] = empty;
					boards[tx+move_idx] = temp;
					move_idx++;
					temp = b;
				}
				if(!(y%2) && y>2 && b.pieces[x][y-1] <= white_king_moved && b.pieces[x][y-1]>0 && !b.pieces[x+1][y-2])
				{
					//TODO add double takes here
					temp.pieces[x+1][y-2] = temp.pieces[x][y];
					temp.pieces[x][y] = empty;
					temp.pieces[x][y-1] = empty;
					boards[tx+move_idx] = temp;
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
					boards[tx+move_idx] = temp;
					move_idx++;
					temp = b;
				}
				if(y%2 && y<7 && !b.pieces[x][y+1])
				{
					temp.pieces[x][y+1] = temp.pieces[x][y];
					temp.pieces[x][y] = empty;
					boards[tx+move_idx] = temp;
					move_idx++;
					temp = b;
				}
				if(!(y%2) && x != 0 && !b.pieces[x-1][y+1])
				{
					temp.pieces[x-1][y+1] = temp.pieces[x][y];
					temp.pieces[x][y] = empty;
					boards[tx+move_idx] = temp;
					move_idx++;
					temp = b;
				}
				if(!(y%2) && !b.pieces[x][y+1])
				{
					temp.pieces[x][y+1] = temp.pieces[x][y];
					temp.pieces[x][y] = empty;
					boards[tx+move_idx] = temp;
					move_idx++;
					temp = b;
				}
				if(y%2 && y<6 && x!= 3 && b.pieces[x+1][y+1] <= white_king_moved && b.pieces[x+1][y+1] > 0 && !b.pieces[x+1][y+2]) 
				{
					//TODO add double takes here
					temp.pieces[x+1][y+2] = temp.pieces[x][y];
					temp.pieces[x][y] = empty;
					temp.pieces[x+1][y+1] = empty;
					boards[tx+move_idx] = temp;
					move_idx++;
					temp = b;
				}
				if(y%2 && y<6 && x != 0 && b.pieces[x][y+1] <= white_king_moved && b.pieces[x][y+1] > 0 && !b.pieces[x-1][y+2])
				{
					//TODO add double takes here
					temp.pieces[x-1][y+2] = temp.pieces[x][y];
					temp.pieces[x][y] = empty;
					temp.pieces[x][y+1] = empty;
					boards[tx+move_idx] = temp;
					move_idx++;
					temp = b;
				}
				if(!(y%2) && y<5 && x != 0 && b.pieces[x-1][y+1] <= white_king_moved && b.pieces[x-1][y+1] > 0 && !b.pieces[x-1][y+2])
				{
					//TODO add double takes here
					temp.pieces[x-1][y+2] = temp.pieces[x][y];
					temp.pieces[x][y] = empty;
					temp.pieces[x-1][y+1] = empty;
					boards[tx+move_idx] = temp;
					move_idx++;
					temp = b;
				}
				if(!(y%2) && y<5 && b.pieces[x][y+1] <= white_king_moved && b.pieces[x][y+1] > 0 && !b.pieces[x+1][y+2])
				{
					//TODO add double takes here
					temp.pieces[x+1][y+2] = temp.pieces[x][y];
					temp.pieces[x][y] = empty;
					temp.pieces[x][y+1] = empty;
					boards[tx+move_idx] = temp;
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
	int moveCount = 1;
	bool drawFlag = false;
        b[0] = bad_board;

        b[0].pieces[1][1] = white_reg; b[0].pieces[1][5] = black_reg;
	makeMove(b);
	printBoard(b[0]);
	//initBoard(b);
/*
	while(1)
	{
		printBoard(b[0]);
		makeMove(b);
		for (int i = 0; i < 256; i++)
			for(int a = 0; a < 4; a++)
				for(int j = 0; j < 8; j++)
					if(b[i].pieces[a][j] != 0)
					{
						//printf("B: %d, Loc: (%d, %d), piece: %d\n", i, a, j, b[i].pieces[a][j]);
						printBoard(b[i]);
						break;
					}
		for(int i = 0; i < moveCount; i++)
		{
			int score = analyseBoard(&b[i], white);
			//printf("B: %d Score: %d\n", i, score);
			if(!analyseBoard(&b[i], black)) 
			{
				printBoard(b[i]);
				return 0;
			}
			//printBoard(b[i]);

		}
	}
		*/
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
				default:
					printf("_|_|");
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
				default:
					printf("_|_|");
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
	
	host_output = new Board[outputSize];
	host_input =  board;
	if(!host_output)
	{
		std::cerr << "operator new failed on size: " 
			<< outputSize << std::endl;
		return -1;
	}
	printf("%d", BLOCK_SIZE);
	#if USE_GPU
	// cuda malloc
	gpuErrChk(cudaMalloc(&device_output, outputSize * sizeof(Board)));
	gpuErrChk(cudaMalloc(&device_input, inputSize * sizeof(Board)));
	
	// cuda memcpy
	gpuErrChk(cudaMemcpy(device_input, host_input, inputSize * sizeof(*device_input),
				cudaMemcpyHostToDevice));

	//launch kernel and check errors
	printf("initializing kernel with block dim: %d and grid dim: %d", 
					(int)ceil(inputSize/(double)BLOCK_SIZE), BLOCK_SIZE);
	dim3 dimBlock((int)ceil(inputSize/(double)BLOCK_SIZE));
	dim3 dimGrid(BLOCK_SIZE);
	expand<<<dimGrid, dimBlock>>>(device_input,
							 	device_output,
							 	inputSize);
	gpuErrChk(cudaPeekAtLastError());
	gpuErrChk(cudaDeviceSynchronize());

	inputSize = outputSize;
	outputSize = inputSize * 512;

	gpuErrChk(cudaFree(device_input));
	device_input = device_output;
	gpuErrChk(cudaMalloc(&device_output, 
				outputSize * sizeof(*device_output)));
	
	printf("initializing kernel with block dim: %d and grid dim: %d", 
					(int)ceil(inputSize/(double)BLOCK_SIZE), BLOCK_SIZE);
	dim3 dimBlock2((int)ceil(inputSize/(double)BLOCK_SIZE));
	expand<<<dimGrid, dimBlock2>>>(device_input,
							 	device_output,
							 	inputSize);
	gpuErrChk(cudaPeekAtLastError());
	gpuErrChk(cudaDeviceSynchronize());
	#endif

	

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
