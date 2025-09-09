#include <stdio.h>

int main()
{
  int T;
  scanf("%d", &T);

  for (int tc = 1; tc <= T; tc++)
  {
    double N;
    scanf("%lf", &N);

    char result[20] = ""; // 최대 12자리 저장 가능
    int idx = 0;
    int overflow = 0;

    while (N > 0)
    {
      if (idx == 12)
      { // 12자리를 넘어가면 overflow
        overflow = 1;
        break;
      }
      N *= 2;
      if (N >= 1)
      {
        result[idx++] = '1';
        N -= 1;
      }
      else
      {
        result[idx++] = '0';
      }
    }
    result[idx] = '\0'; // 문자열 종료

    if (overflow)
      printf("#%d overflow\n", tc);
    else
      printf("#%d %s\n", tc, result);
  }
  return 0;
}