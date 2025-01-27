MODULE kinds
  IMPLICIT NONE
  INTEGER, PARAMETER :: i2b = selected_int_kind(4)
  INTEGER, PARAMETER :: i4b = selected_int_kind(9)
  INTEGER, PARAMETER :: i8b = selected_int_kind(18)
  INTEGER, PARAMETER :: int16 = selected_int_kind(4)
  INTEGER, PARAMETER :: int32 = selected_int_kind(9)
  INTEGER, PARAMETER :: int64 = selected_int_kind(18)

  INTEGER, PARAMETER :: DP = selected_real_kind(14, 200)
  INTEGER, PARAMETER :: float32 = selected_real_kind(6, 37)
  INTEGER, PARAMETER :: float64 = selected_real_kind(14, 200)

  INTEGER, PARAMETER :: CHAR_LEN = 8
END MODULE kinds


