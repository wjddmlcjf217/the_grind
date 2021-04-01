# 268. Missing Number
# Easy

# 2823

# 2485

# Add to List

# Share
# Given an array nums containing n distinct numbers in the range [0, n], return the only number in the range that is missing from the array.

# Follow up: Could you implement a solution using only O(1) extra space complexity and O(n) runtime complexity?


# Example 1:

# Input: nums = [3,0,1]
# Output: 2
# Explanation: n = 3 since there are 3 numbers, so all numbers are in the range [0,3]. 2 is the missing number in the range since it does not appear in nums.
# Example 2:

# Input: nums = [0,1]
# Output: 2
# Explanation: n = 2 since there are 2 numbers, so all numbers are in the range [0,2]. 2 is the missing number in the range since it does not appear in nums.
# Example 3:

# Input: nums = [9,6,4,2,3,5,7,0,1]
# Output: 8
# Explanation: n = 9 since there are 9 numbers, so all numbers are in the range [0,9]. 8 is the missing number in the range since it does not appear in nums.
# Example 4:

# Input: nums = [0]
# Output: 1
# Explanation: n = 1 since there is 1 number, so all numbers are in the range [0,1]. 1 is the missing number in the range since it does not appear in nums.

# Brute Force using set
class Solution:
    def missingNumber(self, nums: List[int]) -> int:
        # numbers in nums is 0 - len(nums)
        # evaluate if each value is in the set, if not return that index 
        # account for the fact the range method is not inclusive, so return i+1 at the end if we go to the end  
        numbers = set(nums)
        for i in range(len(nums)):
            if i not in numbers:
                return i
        return i + 1


# 448. Find All Numbers Disappeared in an Array
# Easy

# 3979

# 287

# Add to List

# Share
# Given an array of integers where 1 ≤ a[i] ≤ n (n = size of array), some elements appear twice and others appear once.

# Find all the elements of [1, n] inclusive that do not appear in this array.

# Could you do it without extra space and in O(n) runtime? You may assume the returned list does not count as extra space.

# Example:

# Input:
# [4,3,2,7,8,2,3,1]

# Output:
# [5,6]

# brute force using hashmap to keep track of numbers that exist within possible range, then returning values that never got incremented for not existing
class Solution:
    def findDisappearedNumbers(self, nums: List[int]) -> List[int]:
        output = []
        if not nums:
            return output
        num_map = {}
        max_num = len(nums)
        for i in range(1, max_num + 1):
            num_map[i] = 0
        for i in range(len(nums)):
            if nums[i] in num_map:
                num_map[nums[i]] += 1
        for key, value in num_map.items():
            if value == 0:
                output.append(key)
        return output

# using the nums[nums[i]%n]+= n expresssion to find missing numbers, hashing method using index 
class Solution:
    def findDisappearedNumbers(self, nums: List[int]) -> List[int]:
        output = []
        if not nums:
            return output
        # range of possible values is 1 - len(nums) 
        nums.append(0)
        n = len(nums)
        for i in range(n):
            nums[nums[i]%n] += n
        for i in range(n):
            if nums[i] // n == 0:
                output.append(i)
        return output

# 41. First Missing Positive
# Hard

# 5390

# 953

# Add to List

# Share
# Given an unsorted integer array nums, find the smallest missing positive integer.

 

# Example 1:

# Input: nums = [1,2,0]
# Output: 3
# Example 2:

# Input: nums = [3,4,-1,1]
# Output: 2
# Example 3:

# Input: nums = [7,8,9,11,12]
# Output: 1

# after removing all the numbers greater than or equal to n, all the numbers remaining are smaller than n. If any number i appears, 
# we add n to nums[i] which makes nums[i]>=n. Therefore, if nums[i]<n, it means i never appears in the array and we should return i.
class Solution:
    def firstMissingPositive(self, nums: List[int]) -> int:
        # for any array whose length is l, the first missing
        # positive must be in range [1,...,l+1], eg. [1, 2, 3] -> 4
        # so we only have to care about those elements in this range and
        # remove the rest.
        # 2. we can use the array index as the hash to restore the 
        # frequency of each number within  the range [1,...,l+1] 
        
        nums.append(0) # in order to hash, we need a 0 value
        n = len(nums) # original n + 1, since we added 0 above
        for i in range(len(nums)):
            if nums[i] < 1 or nums[i] >= n: 
                nums[i] = 0
        for i in range(len(nums)):
            nums[nums[i]%n] += n
        for i in range(1, len(nums)):
            if nums[i] // n == 0:
                return i
        return n

# nlog(n) solution by sorting first
class Solution:
    def firstMissingPositive(self, nums: List[int]) -> int:
        nums.sort()
        output = 1
        for i in range(len(nums)):
            if nums[i] == output:
                output += 1
        return output

# Q: Given a non-empty array of integers, every element appears twice except for one. Find that single one.
# Your algorithm should have a linear runtime complexity. Could you implement it without using extra memory?

# Example 1:
# Input: [2,2,1]
# Output: 1

# Example 2:
# Input: [4,1,2,1,2]
# Output: 4


# O(n^2) since iterate through nums O(n), then search method is also O(n), thus O(n^2)
def single_number(nums):
    no_duplicate_list = []
    for num in nums:
        if i not in no_duplicate_list:
            no_duplicate_list.append(num)
        else:
            no_duplicate_list.remove(num)
    return no_duplicate_list.pop()

# O(n) using hashmap, assigning each number in nums as key value pairs in hashmap, then increment it by 1 if you encounter a duplicate, finally return the key with the value that == 1 
def singleNumber(self, nums: List[int]) -> int:
    number_tracker = {}
    for i in range(len(nums)):
        number_tracker[nums[i]] = 0

    for i in range(len(nums)):
        if nums[i] in number_tracker:
            number_tracker[nums[i]] += 1
    
    for key, val in number_tracker.items():
        if val == 1:
            return key

# Optimal way, O(n) since iterating thorugh nums is O(n), and hashmap operations like pop is O(1), O(n * 1) = O(n)
def single_number(nums):
    hash_map = {}
    for i in nums:
        try:
            # preemptively try to pop element even if its not in it, this checks for the dupe later the except clause
            hash_map.pop(i)
        except:
            # when the try statement fails, assign key value to hashmap
            hash_map[i] = 1
    return hash_map.popitem()[0]

# Q: You're given strings J representing the types of stones that are jewels, and S representing the stones you have.  Each character in S is a type of stone you have.  You want to know how many of the stones you have are also jewels.

# The letters in J are guaranteed distinct, and all characters in J and S are letters. Letters are case sensitive, so "a" is considered a different type of stone from "A".

# Example 1:

# Input: J = "aA", S = "aAAbbbb"
# Output: 3
# Example 2:

# Input: J = "z", S = "ZZ"
# Output: 0
# Note:

# S and J will consist of letters and have length at most 50.
# The characters in J are distinct.

# Hashmap solution
def numJinS(J, S):
    hashmap = {}
    for char in J:
        hashmap[char] = 0
        if char in S:
            hashmap[char] += S.count(char)
    return sum(hashmap.values())

# Set solution
def numJinS(J, S):
    count = 0
    for i in range(len(S)):
        if S[i] in set(J):
            count += 1
    return count

# Q: Given an array of integers, return indices of the two numbers such that they add up to a specific target.

# You may assume that each input would have exactly one solution, and you may not use the same element twice.

# Example:

# Given nums = [2, 7, 11, 15], target = 9,

# Because nums[0] + nums[1] = 2 + 7 = 9,
# return [0, 1].

# dumb bruteforce O(n^2)
def bruteTwosums(num, target):
    for i in range(len(num)):
        for j in range(i+1, len(num)):
            sum = num[i] + num[j]
            if sum == target:
                return [i, j]

# Optimal hashmap solution O(n) since we're searching once O(n) then hashmap methods are O(1) thus O(n * 1)
def twosums(num, target): # [2, 7, 10], target = 9
    hashmap = {} # {2: 0, }
    for i, n in enumerate(num): # enum assigns a number to each element, so we can track index 
        complement = target - n # 7 = 9 - 2, 2 = 9 - 7
        if complement in hashmap:
            return [hashmap[complement], i]
        else:
            hashmap[n] = i
    
# 2 pointer method assuming input array is sorted in ascending order
def twoSum(numbers: List[int], target: int) -> List[int]:
    left, right = 0, len(numbers) - 1
    while left <= right:
        sum = numbers[left]  + numbers[right]
        if sum == target:
            return [left + 1, right + 1]
        elif sum > target:
            right -= 1
        else:
            left += 1

# 977. Squares of a Sorted Array
# Easy

# 2146

# 114

# Add to List

# Share
# Given an integer array nums sorted in non-decreasing order, return an array of the squares of each number sorted in non-decreasing order.

# Example 1:

# Input: nums = [-4,-1,0,3,10]
# Output: [0,1,9,16,100]
# Explanation: After squaring, the array becomes [16,1,0,9,100].
# After sorting, it becomes [0,1,9,16,100].
# Example 2:

# Input: nums = [-7,-3,2,3,11]
# Output: [4,9,9,49,121]
2 pointer
class Solution:
    def sortedSquares(self, nums: List[int]) -> List[int]:
        output = deque([])
        left = 0
        right = len(nums) - 1
        while left <= right:
            if abs(nums[left]) > abs(nums[right]):
                output.appendleft(nums[left]*nums[left])
                left += 1
            else:
                output.appendleft(nums[right]*nums[right])
                right -= 1
        return output

# Q: Given an array of integers, find if the array contains any duplicates.

# Your function should return true if any value appears at least twice in the array, and it should return false if every element is distinct.

# Example 1:

# Input: [1,2,3,1]
# Output: true
# Example 2:

# Input: [1,2,3,4]
# Output: false

# Stupid O(n^2) way
def containsDuplicates(nums):
    for i in range(len(nums)):
        for j in range(i+1, len(nums)):
            if nums[i] == nums[j]:
                 return True
    return False

# Hashmap or set O(n) way
def containsDuplicates(nums):
    hash = {}
    for i in range(len(nums)):
        if nums[i] not in hash:
            hash[nums[i]] = i
        else:
            return True
    return False

    dupes = set()
    for i in range(len(nums)):
        if nums[i] not in dupes:
            dupes.add(nums[i])
        else:
            return True
    return False


# Q: Say you have an array for which the ith element is the price of a given stock on day i.

# If you were only permitted to complete at most one transaction (i.e., buy one and sell one share of the stock), design an algorithm to find the maximum profit.

# Note that you cannot sell a stock before you buy one.

# Example 1:

# Input: [7,1,5,3,6,4]
# Output: 5
# Explanation: Buy on day 2 (price = 1) and sell on day 5 (price = 6), profit = 6-1 = 5.
#              Not 7-1 = 6, as selling price needs to be larger than buying price.
# Example 2:

# Input: [7,6,4,3,1]
# Output: 0
# Explanation: In this case, no transaction is done, i.e. max profit = 0.

# Brute force O(n^2)
def maxprofit(prices):
    min_price = float('inf')
    max_profit = 0
    for i in range(len(prices)):
        min_price = prices[i]
        for j in range(i, len(prices)):
            if prices[j] < min_price:
                min_price = prices[j]
            if prices[j] - min_price > max_profit:
                max_profit = prices[j] - min_price
    return max_profit

# O(n) method, always remember to set min to big number and max to small number in these types of qs
def maxprofit(prices):
    min_price = float('inf')
    max_profit = 0 
    for i in range(len(prices)):
        if prices[i] < min_price:
            min_price = prices[i]
        if prices[i] - min_price > max_profit:
            max_profit = prices[i] - min_price
    return max_profit

# Q:Given an array nums of n integers where n > 1,  return an array output such that output[i] is equal to the product of all the elements of nums except nums[i].

# Example:

# Input:  [1,2,3,4]
# Output: [24,12,8,6]
# Note: Please solve it without division and in O(n).

# Follow up:
# Could you solve it with constant space complexity? (The output array does not count as extra space for the purpose of space complexity analysis.)

def productExceptSelf(nums: List[int]) -> List[int]:
    
    output, L, R = [0]*len(nums), [0]*len(nums), [0]*len(nums)
    L[0] = 1
    R[len(nums) - 1] = 1
    
    for i in range(1, len(nums)):
        L[i] = nums[i-1] * L[i-1]
    
    for i in reversed(range(len(nums) - 1)):
        R[i] = nums[i+1] * R[i+1]
    
    for i in range(len(nums)):
        output[i] = L[i] * R[i]
    
    return output

# Constant space complexity by not using additional arrays but rather using variable R to keep track of right prodcuts
def productExceptSelf(nums: List[int]) -> List[int]:
        
    output = [0]*len(nums)
    output[0] = 1
    
    for i in range(1, len(nums)):
        output[i] = nums[i-1] * output[i-1]
    
    R = 1
    for i in reversed(range(len(nums))):
        output[i] = R * output[i]
        R = R * nums[i]
    
    return output

# Q: Given an integer array nums, find the contiguous subarray (containing at least one number) which has the largest sum and return its sum.

# Example:

# Input: [-2,1,-3,4,-1,2,1,-5,4],
# Output: 6
# Explanation: [4,-1,2,1] has the largest sum = 6.
# Follow up:

# If you have figured out the O(n) solution, try coding another solution using the divide and conquer approach, which is more subtle.

# brute force
def maxSubArray(self, nums: List[int]) -> int:
    # brute force, have a helper array to keep track of sums
    # take every combination from the array to account for sum
    # return the greatest value in the helper array 
    
    sums_arr = []
    for i in range(len(nums)):
        sums_arr.append(nums[i])
        curr_sum = nums[i]
        for j in range(i+1, len(nums)):
            curr_sum += nums[j]
            sums_arr.append(curr_sum)
    return max(sums_arr)

def maxSubArray(self, nums: List[int]) -> int:
    # the greatest contiguous value is going to be the current value
    # or the current value + the previous values
    # compare the current value vs current value + previous values
    # update the maximum contiguous variable and return it 
    
    max_sum = nums[0]
    max_curr = nums[0]
    
    for i in range(1, len(nums)):
        max_curr += nums[i]
        if nums[i] > max_curr:
            max_curr = nums[i]
        if max_curr > max_sum:
            max_sum = max_curr
        return max_sum

def maxSubArray(nums: List[int]) -> int:
    
    sums = [0]*len(nums)
    sums[0] = nums[0]
    
    for i in range(1, len(nums)):
        sum = nums[i] + sums[i-1]
        if sum > nums[i]:
            sums[i] = sum
        else:
            sums[i] = nums[i]
    
    return max(sums)
    
# Kadane algo: greatest subarray is either the ith element or the ith element + previous elements
def maxSubArray(nums: List[int]) -> int:

    max_curr = max_sum = nums[0]
    for i in range(1, len(nums)):
        max_curr = max(nums[i], nums[i] + max_curr)
        if max_curr > max_sum:
            max_sum = max_curr
    return max_sum
# Q: Given an integer array nums, find the contiguous subarray within an array (containing at least one number) which has the largest product.

# Example 1:

# Input: [2,3,-2,4]
# Output: 6
# Explanation: [2,3] has the largest product 6.
# Example 2:

# Input: [-2,0,-1]
# Output: 0
# Explanation: The result cannot be 2, because [-2,-1] is not a subarray.

def maxProduct(self, nums: List[int]) -> int:
    # brute force, have a prod_arr to store every combination of values 
    # return the greatest value from prod_arr
    
    prod_arr = []
    for i in range(len(nums)):
        prod_arr.append(nums[i])
        curr_prod = nums[i]
        for j in range(i+1, len(nums)):
            curr_prod *= nums[j]
            prod_arr.append(curr_prod)
    return max(prod_arr)

def maxProduct(nums: List[int]) -> int:
    prev_max = nums[0]
    prev_min = nums[0]
    ans = nums[0]

    for i in range(1, len(nums)):
        curr_max = max(prev_max*nums[i], prev_min*nums[i], nums[i]) # compares 3 possible maximum products
        curr_min = min(prev_max*nums[i], prev_min*nums[i], nums[i]) # becaue of negative integers, we need to keep track of minimum value in case 
                                                                    # we encounter another negative to make it positive and thus becoming greatest product, also takes care of 0
        ans = max(curr_max, ans) # max product will be either the current max (ans) or the newest max (curr_max)
        prev_max = curr_max
        prev_min = curr_min # updates negative value in case we encounter another negative value, also accounts for 0 
        
    return ans


# 189. Rotate Array
# Medium

# 4121

# 904

# Add to List

# Share
# Given an array, rotate the array to the right by k steps, where k is non-negative.

# Follow up:

# Try to come up as many solutions as you can, there are at least 3 different ways to solve this problem.
# Could you do it in-place with O(1) extra space?
 

# Example 1:

# Input: nums = [1,2,3,4,5,6,7], k = 3
# Output: [5,6,7,1,2,3,4]
# Explanation:
# rotate 1 steps to the right: [7,1,2,3,4,5,6]
# rotate 2 steps to the right: [6,7,1,2,3,4,5]
# rotate 3 steps to the right: [5,6,7,1,2,3,4]

# brute force O(n^2) since insert and for loop is O(n)
def rotate(self, nums: List[int], k: int) -> None:
    """
    Do not return anything, modify nums in-place instead.
    """

    for i in range(k):
        popped_val = nums.pop()
        nums.insert(0, popped_val)

# O(n) solution using additional array
def rotate(self, nums: List[int], k: int) -> None:
    """
    Do not return anything, modify nums in-place instead.
    """

    rotated_arr = [0]*len(nums)
    for i in range(len(nums)):
        # Given nums= [1,2,3,4,5,6,7], k = 3
        # The expression (i+k) % len(nums) each iteration = 3,4,5,6,0,1,2
        rotated_arr[(i+k)%len(nums)] = nums[i]
    nums[:] = rotated_arr # list splice expression to make sure nums array is replaced by shifted_arr, without it wouldn't work

BINARY SEARCH
# 704. Binary Search
# Easy

# 1221

# 58

# Add to List

# Share
# Given an array of integers nums which is sorted in ascending order, and an integer target, write a function to search target in nums. If target exists, then return its index. Otherwise, return -1.

 

# Example 1:

# Input: nums = [-1,0,3,5,9,12], target = 9
# Output: 4
# Explanation: 9 exists in nums and its index is 4
# Example 2:

# Input: nums = [-1,0,3,5,9,12], target = 2
# Output: -1
# Explanation: 2 does not exist in nums so return -1

class Solution:
    def search(self, nums: List[int], target: int) -> int:
        left = 0
        right = len(nums) - 1
        
        while left <= right:
            mid = left + (right - left) // 2
            if nums[mid] == target:
                return mid
            elif nums[mid] > target:
                right = mid - 1
            elif nums[mid] < target:
                left = mid + 1
        return -1



# Q: Suppose an array sorted in ascending order is rotated at some pivot unknown to you beforehand.

# (i.e.,  [0,1,2,4,5,6,7] might become  [4,5,6,7,0,1,2]).

# Find the minimum element.

# You may assume no duplicate exists in the array.

# Example 1:

# Input: [3,4,5,1,2] 
# Output: 1
# Example 2:

# Input: [4,5,6,7,0,1,2]
# Output: 0

# brute force just doing linear search for minimum value, O(n)
def findMin(self, nums: List[int]) -> int:
    min_val = float('inf')
    for i in range(len(nums)):
        if nums[i] < min_val:
            min_val = nums[i]
    return min_val


def findMin(self, nums: List[int]) -> int:
    # binary search to find the smallest value in rotated array
    # [4,5,6,7,0,1,2], set right bound to "mid" if right boudn > mid
    # set left bound to mid left is greater than mid
    # imemdiately check left of mid, if it is greater than mid return mid
    # check right of mid, if is smaller than mid, return right
    # if right bound is greater than left, then its already in ascending order
    left = 0
    right = len(nums) - 1
    if len(nums) == 1:
        return nums[0]
    while left < right:
        mid = left + (right - left) // 2
        if nums[mid-1] > nums[mid]:
            return nums[mid]
        elif nums[mid+1] < nums[mid]:
            return nums[mid+1]
        elif nums[right] > nums[left]:
            return nums[left]
        elif nums[right] > nums[mid]:
            right = mid
        elif nums[left] < nums[mid]:
            left = mid

# more elegant solution than above with less comparisons
def findMin(self, nums: List[int]) -> int:
    left = 0
    right = len(nums) - 1
    if len(nums) == 1:
        return nums[0]
    while left < right:
        mid = left + (right - left) // 2
        if nums[mid] > nums[right]:
            left = mid + 1
        else:
            right = mid
    return nums[left]

# 154. Find Minimum in Rotated Sorted Array II
# Hard

# 1409

# 266

# Add to List

# Share
# Suppose an array sorted in ascending order is rotated at some pivot unknown to you beforehand.

# (i.e.,  [0,1,2,4,5,6,7] might become  [4,5,6,7,0,1,2]).

# Find the minimum element.

# The array may contain duplicates.

# Example 1:

# Input: [1,3,5]
# Output: 1
# Example 2:

# Input: [2,2,2,0,1]
# Output: 0

def findMin(self, nums: List[int]) -> int:
    low = 0
    high = len(nums) - 1
    while low < high:
        mid = low + (high - low) // 2
        if nums[high] > nums[low]:
            return nums[low]
        if nums[mid] > nums[high]:
            low = mid + 1
        elif nums[mid] < nums[high]:
            high = mid 
        else:
            high-=1 # this covers the duplicates
    return nums[low]

# Q: Suppose an array sorted in ascending order is rotated at some pivot unknown to you beforehand.

# (i.e., [0,1,2,4,5,6,7] might become [4,5,6,7,0,1,2]).

# You are given a target value to search. If found in the array return its index, otherwise return -1.

# You may assume no duplicate exists in the array.

# Your algorithm's runtime complexity must be in the order of O(log n).

# Example 1:

# Input: nums = [4,5,6,7,0,1,2], target = 0
# Output: 4
# Example 2:

# Input: nums = [4,5,6,7,0,1,2], target = 3
# Output: -1

def search(nums: List[int], target: int) -> int:
    
    if target not in nums or len(nums) == 0:
        return -1

    low, high = 0, len(nums) - 1

    while low <= high:
        mid = low + (high - low) // 2
        if target == nums[mid]:
            return mid

        if nums[low] <= nums[mid]:
            if nums[low] <= target <= nums[mid]:
                high = mid - 1
            else:
                low = mid + 1
        else:
            if nums[mid] <= target <= nums[high]:
                low = mid + 1
            else:
                high = mid - 1

# Given n non-negative integers a1, a2, ..., an , where each represents a point at coordinate (i, ai). n vertical lines are drawn such that the two endpoints of line i is at (i, ai) and (i, 0). Find two lines, which together with x-axis forms a container, such that the container contains the most water.

# Note: You may not slant the container and n is at least 2.

# Example:

# Input: [1,8,6,2,5,4,8,3,7]
# Output: 49

# brute force O(n^2)
def maxArea(self, height: List[int]) -> int:
        max_Area = 0
        for i in range(len(height)):
            for j in range(len(height)):
                area = (j - i) * (min(height[i], height[j]))
                if area > max_Area:
                    max_Area = area
        return max_Area

# length is the difference of 2 pointers, while the width or height is the actual values of pointer index in array
# area will only get bigger if the height is greater so increment low when height[low] < height[high]
# O(n) linear complexity 
def maxArea(self, height: List[int]) -> int:
        low = 0
        high = len(height) - 1
        max_Area = 0
        
        while low < high:
            area = (high - low) * min(height[low], height[high])
            if area > max_Area:
                max_Area = area
            if height[low] < height[high]: # because area will only increase if the height[] values are bigger, if current height[left] is lower then try the next 
                low += 1
            else:
                high -= 1
        return max_Area

# // Given two arrays of integers, find a pair of values (one from each array) that you can swap to give the two arrays the same value.
# // EXAMPLE
# // IN : [4,1,2,1,1,2] ,  [3,6,3,3]
# // OUT: [1,3]
# original sum of arr1: 11
# original sum of arr2: 15
# final sum(arr1): 11 - 1 + 3 = 13,    ++list1_sum - list1[i] + list2[j] = list2_sum - list2[j] + list1[i]

# final sum(arr2): 15 - 3 + 1 = 13     ++list2_sum - list2[j] + list1[i] = final_list2_sum = 13

# keep track of the intial sums of both arrays
# process the array 1 swap element in arr1 with elements in arr2 and calculate the sum for both arrays
# if the sums of the swapped arrays equal each other then i can just return the pair of the swapped integers
# consider an empty arrays, consider no sum match 
# [3,2], [1,2] => -[2, 1]

def swap_pair(list1, list2): 
   list1_sum = sum(list1)
   list2_sum = sum(list2)
   
   
   for i in range(len(list1)): # [i] = 2
      
       for j in range(len(list2)): # [j] = 1
       
           if list1_sum - list1[i] + list2[j] == list2_sum - list2[j] + list1[i]: # 5 - 2 + 1 == 3 - 1  + 2, 4 == 4         
               return [list1[i], list2[j]] 
           
# 867. Transpose Matrix
# Easy

# 585

# 328

# Add to List

# Share
# Given a 2D integer array matrix, return the transpose of matrix.

# The transpose of a matrix is the matrix flipped over its main diagonal, switching the matrix's row and column indices.


# Example 1:

# Input: matrix = [[1,2,3],[4,5,6],[7,8,9]]
# Output: [[1,4,7],[2,5,8],[3,6,9]]
# Example 2:

# Input: matrix = [[1,2,3],[4,5,6]]
# Output: [[1,4],[2,5],[3,6]]

def transpose(self, matrix: List[List[int]]) -> List[List[int]]:
    converted_arr = []
    columns = len(matrix)
    rows = len(matrix[0])
    
    for i in range(rows):
        inner_arr = []
        for j in range(columns):
            inner_arr.append(matrix[j][i])
        converted_arr.append(inner_arr)
    return converted_arr




# concur question
# bruteforce
def fizzBuzz(self, n: int) -> List[str]:
    output = []
    for i in range(1, n+1):
        # starting in descending order 15-3 as starting up could skip over potential FizzBuzz since 15 is divisible by 3 and 5
        if i % 15 == 0:
            output.append("FizzBuzz")
        elif i % 5 == 0:
            output.append("Buzz")
        elif i % 3 == 0:
            output.append("Fizz")
        else:
            output.append(str(i))
    return output
        
# hashmap + concatenation way
def fizzBuzz(self, n: int) -> List[str]:
    output = []
    d = {3: "Fizz", 5: "Buzz"}
    for i in range(1, n+1):
        helper = ""
        for k,v in d.items():
            if i % k == 0:
                helper += v
        if not helper:
            helper += str(i)
        output.append(helper)
        
    return output

def myAtoi(self, str: str) -> int:
    output = 0
    # strip leading/trailing white spaces
    str = str.strip()
    # handle empty string
    if len(str) == 0:
        return output
    # check if str begins with an alphabet
    if str[0].isalpha():
        return output
    # handle negative sign
    is_negative = False
    if str[0] == "-":
        is_negative = True
    # establish upper and lower limits
    upper_bound = 2147483648 - 1
    lower_bound = -2147483648
    # establish valid characters
    valid_char = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

    # check for - or + signs in the beginning 
    if str[0] == "-" or str[0] == "+":
        str = str[1:]
    # proces the string and build the output
    for char in str:
        if char not in valid_char:
            break
        else:
            output = output * 10 + int(char)
        # The way to build output using ascii
        # for i in range(len(str)):
        # if str[i] not in valid_char:
        #     break
        # else:
        #     output = output * 10  + (ord(str[i]) - ord("0"))
    # check if negative
    if is_negative:
        output = -output
    # check for upper and lower bounds
    if output > upper_bound:
        return upper_bound
    if output < lower_bound:
        return lower_bound
    # return output at the end
    return output

Recursion
def fib(self, N: int) -> int:
    if N == 1 or N == 2:
        return 1
    else:
        return fib(N-1) + fib(N-2)

Dynamic Programming
def fib(self, N: int) -> int:
    first, second, answer = 1, 1, 1
    if N == 1 or N == 2:
        return 1
    if N == 0:
        return 0
    for i in range(3, N+1):
        answer = first + second
        first = second
        second = answer
    return answer


# 70. Climbing Stairs
# Easy

# 5859

# 185

# Add to List

# Share
# You are climbing a staircase. It takes n steps to reach the top.

# Each time you can either climb 1 or 2 steps. In how many distinct ways can you climb to the top?

 

# Example 1:

# Input: n = 2
# Output: 2
# Explanation: There are two ways to climb to the top.
# 1. 1 step + 1 step
# 2. 2 steps
# Example 2:

# Input: n = 3
# Output: 3
# Explanation: There are three ways to climb to the top.
# 1. 1 step + 1 step + 1 step
# 2. 1 step + 2 steps
# 3. 2 steps + 1 step

def climbStairs(self, n: int) -> int:

    if n == 0:
        return 0
    if n == 1:
        return 1
    
    stair_count = [0]*(n+1)
    stair_count[0] = 0
    stair_count[1] = 1
    stair_count[2] = 2
    
    for i in range(3, n+1):
        stair_count[i] = stair_count[i-1]  + stair_count[i-2] # since you can only take 1 or 2 steps, to get to nth step it will always be (n-1) + (n-2) steps 
    
    return stair_count[n]


# coin change 
# given n cents, find out minimum amount of coins to give back
# if n = 50, then output = [25, 25], 2 x 25 cents is the least amount of coins
# this is a greedy solution, doesn't work for everything
def coin_change(n):
    output = []
    denominations = [25, 10, 5, 1]

    i = 0
    upper_bound = len(denominations) - 1
    
    # traversing through all denominations
    while i <= upper_bound:
        # validate that the current n is greater than the denominations and find correct denomination
        while n >= denominations[i]:
            n -= denominations[i]
            output.append(denominations[i])

        i += 1

    return output
    


# Q: Given a string, determine if it is a palindrome, considering only alphanumeric characters and ignoring cases.

# Note: For the purpose of this problem, we define empty string as valid palindrome.

# Example 1:

# Input: "A man, a plan, a canal: Panama"
# Output: true
# Example 2:

# Input: "race a car"
# Output: false

# .isalnum() validates only alphanumeric characters, useful getting rid of whitespace, special characters, etc
def isPalindrome(s: str) -> bool:
    if len(s) == 0:
        return True
    new_str = "".join(char for char in s if char.isalnum()).lower()
    
    left, right = 0, len(new_str) - 1
    
    while left < right:
        if new_str[left] != new_str[right]:
            return False
        else:
            left += 1
            right -= 1
    return True

# brute force
# check if all permutations of a given string is a palindrome

def permutaiton_is_palindrome(s:str) -> bool:
    # use 2 for loops to compare string values and keep a track of char occurences
    # if char count is even then it's fine as that's valid to make a palindrome
    # if char count is odd then increment an odd counter, a valid palindrome only one character can have odd match

    odd_counter = 0

    for i in range(len(str)):
        char_counter = 1
        for j in range(len(str)):
            if str[i] == str[j]:
                char_counter += 1
        if char_counter % 2 != 0:
            odd_counter += 1
    
    if odd_counter > 1:
        return False
    return True
    
# optimal solution
# check if all permutations of a given string is a palindrome

def permutaiton_is_palindrome(s:str) -> bool:
    # 256 characters for ASCII
    total_char = 256

	# Create a count array and initialize 
	# all values as 0 
    char_array = [0]*total_char

	# For each character in input strings, 
	# increment count in the corresponding 
	# count array 
    for i in range(len(str)):
        char_array[ord(str[i])] += 1
    
	# Count odd occurring characters 
    odd = 0

    for i in range(len(char_array)):
        if char_array[i] % 2 != 0:
            odd += 1
        # if odd exceeds 1 then that breaks our logic and is no longer palindrome
        if odd > 1:
            return False

	# Return true if odd count is 0 or 1, 
    return True
        

# 394. Decode String

# Given an encoded string, return its decoded string.

# The encoding rule is: k[encoded_string], where the encoded_string inside the square brackets is being repeated exactly k times. Note that k is guaranteed to be a positive integer.

# You may assume that the input string is always valid; No extra white spaces, square brackets are well-formed, etc.

# Furthermore, you may assume that the original data does not contain any digits and that digits are only for those repeat numbers, k. For example, there won't be input like 3a or 2[4].

# Example 1:

# Input: s = "3[a]2[bc]"
# Output: "aaabcbc"
# Example 2:

# Input: s = "3[a2[c]]"
# Output: "accaccacc"

def decodeString(self, s: str) -> str:
    # 4 different cases: digit, [, alphabet, ] 
    # digit: keep track of all digit in stack, pop it out when we encounter ] 
    # [: preemptively add output string to stack, reset it
    # letter: concatenate letter to output string
    # ]: pop from both stacks, and concatenate whatever value x amount of times
    
    digit_stack = []
    letter_stack = []
    index = 0
    output = ""
    
    while index < len(s):
        if s[index].isdigit():
            number = 0
            while s[index].isdigit():
                number = number * 10 + int(s[index])
                index += 1
            digit_stack.append(number)
        elif s[index] == "[":
            letter_stack.append(output)
            output = ""
            index += 1
        elif s[index].isalpha():
            output +=  s[index]
            index += 1
        elif s[index] == "]":
            times_repeated = digit_stack.pop()
            tmp_chars = letter_stack.pop()
            for i in range(times_repeated):
                tmp_chars += output
            output = tmp_chars
            index += 1
    return output


# for the above question but without [], so 3a2b -> aaabb
        digit_stack = []
        letter_stack = []
        index = 0
        output = ""        
    while index < len(s):
        if s[index].isdigit():
            total_digit = 0
            while s[index].isdigit():
                total_digit = total_digit*10 + int(s[index])
                index += 1
            digit_stack.append(total_digit)
        elif s[index].isalpha():
            total_letter = ""
            while index< len(s) and s[index].isalpha():
                total_letter += s[index]
                index += 1
            count = digit_stack.pop()
            for i in range(count):
                output = output + total_letter
    return output
                        

# 674. Longest Continuous Increasing Subsequence

# Share
# Given an unsorted array of integers, find the length of longest continuous increasing subsequence (subarray).

# Example 1:
# Input: [1,3,5,4,7]
# Output: 3
# Explanation: The longest continuous increasing subsequence is [1,3,5], its length is 3. 
# Even though [1,3,5,7] is also an increasing subsequence, it's not a continuous one where 5 and 7 are separated by 4. 
# Example 2:
# Input: [2,2,2,2,2]
# Output: 1
# Explanation: The longest continuous increasing subsequence is [2], its length is 1. 
SLIDING WINDOW 
def findLengthOfLCIS(self, nums: List[int]) -> int:
    # keep track of anchor and outer bound
    # change anchor when the previous value > current value
    # the result will be the max between whatever result and new result
    
    anchor = 0
    result = 0
    
    for i in range(len(nums)):
        if i > 0 and nums[i - 1] >= nums[i]:
            anchor = i
        result = max(result, i - anchor + 1)
    return result

class Solution:
    def findLengthOfLCIS(self, nums: List[int]) -> int:
        anchor = 0
        lead = 0
        result = 0
        
        while lead < len(nums):
            if lead > 0 and nums[lead] <= nums[lead - 1]:
                anchor = lead
            result = max(result, lead - anchor + 1)
            lead += 1
        return result
# 3. Longest Substring Without Repeating Characters

# Given a string, find the length of the longest substring without repeating characters.

# Example 1:

# Input: "abcabcbb"
# Output: 3 
# Explanation: The answer is "abc", with the length of 3. 
# Example 2:

# Input: "bbbbb"
# Output: 1
# Explanation: The answer is "b", with the length of 1.
SLIDING WINDOW 
def lengthOfLongestSubstring(self, s: str) -> int:
    # have 2 pointers anchor, and lead 
    # check if char at lead is in a set. if not, then add, if it is then remove
    # increemnt the 2 pointers 
    
    anchor = 0
    lead = 0
    output = 0
    unique_char = set()
    
    while lead < len(s):
        if s[lead] not in unique_char:
            unique_char.add(s[lead]) 
            lead += 1
            output = max(output, len(unique_char))
            
        else:
            unique_char.remove(s[anchor]) # remove the last non-unique char we saw
            anchor += 1

    return output

# 209. Minimum Size Subarray Sum
# Medium

# 3492

# 138

# Add to List

# Share
# Given an array of positive integers nums and a positive integer target, return the minimal length of a contiguous subarray [numsl, numsl+1, ..., numsr-1, numsr] of which the sum is greater than or equal to target. If there is no such subarray, return 0 instead.

 

# Example 1:

# Input: target = 7, nums = [2,3,1,2,4,3]
# Output: 2
# Explanation: The subarray [4,3] has the minimal length under the problem constraint.
# Example 2:

# Input: target = 4, nums = [1,4,4]
# Output: 1
# Example 3:

# Input: target = 11, nums = [1,1,1,1,1,1,1,1]
# Output: 0

# brute force
class Solution:
    def minSubArrayLen(self, target: int, nums: List[int]) -> int:
        min_distance = float('inf')
        # edge case when none of the values >= target
        if sum(nums) < target:
            return 0
        for i in range(len(nums)):
            # edge case when a single value >= target
            if nums[i] >= target:
                return 1
            curr_sum = nums[i]
            for j in range(i+1, len(nums)):
                curr_sum += nums[j]
                if curr_sum >= target:
                    min_distance = min(min_distance, j - i + 1)
        return min_distance

SLIDING WINDOW 
# adding up all values and decrementing from the left side until curr_sum < target
# using sliding window, we can calculate distance (i - left + 1) and find min value 
class Solution:
    def minSubArrayLen(self, target: int, nums: List[int]) -> int:
        min_distance = float('inf')
        curr_sum = 0
        left = 0
        
        # edge case when none of values are >= target
        if sum(nums) < target:
            return 0
        
        for i in range(len(nums)):
            # when a single element >= target
            if nums[i] >= target:
                return 1
            curr_sum += nums[i]
            while curr_sum >= target:
                min_distance = min(min_distance, i - left + 1)
                # decrementing left values until we get curr_sum < target
                curr_sum -= nums[left]
                left += 1
        return min_distance

INTERVAL
# 56. Merge Intervals
# Medium

# 6754

# 364

# Add to List

# Share
# Given an array of intervals where intervals[i] = [starti, endi], merge all overlapping intervals, and return an array of the non-overlapping intervals that cover all the intervals in the input.

 

# Example 1:

# Input: intervals = [[1,3],[2,6],[8,10],[15,18]]
# Output: [[1,6],[8,10],[15,18]]
# Explanation: Since intervals [1,3] and [2,6] overlaps, merge them into [1,6].
# Example 2:

# Input: intervals = [[1,4],[4,5]]
# Output: [[1,5]]
# Explanation: Intervals [1,4] and [4,5] are considered overlapping.

class Solution:
    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        output = []
        # lambda x:x[0] sorts the intervals by the first value in each subarray
        for interval in sorted(intervals, key=lambda x: x[0]):
            # [[1,3],[2,6],[8,10],[15,18]], compare [1, 3] and [2, 6]
            # since 2 < 3, there is overlap so we make [1,3] -> [1,6]
            if output and interval[0] <= output[-1][1]:
                # max(interval[1], output[-1][1]) since the 2nd value in 
                # the previous interval may be greater eg. [1, 7] vs [2, 6]
                output[-1][1] = max(interval[1], output[-1][1])
            else:
                # only add interval IF there is a merge except first one
                output.append(interval)
        return output

INTERVAL  
# 986. Interval List Intersections
# Medium

# 2105

# 59

# Add to List

# Share
# You are given two lists of closed intervals, firstList and secondList, where firstList[i] = [starti, endi] and secondList[j] = [startj, endj]. Each list of intervals is pairwise disjoint and in sorted order.

# Return the intersection of these two interval lists.

# A closed interval [a, b] (with a < b) denotes the set of real numbers x with a <= x <= b.

# The intersection of two closed intervals is a set of real numbers that are either empty or represented as a closed interval. For example, the intersection of [1, 3] and [2, 4] is [2, 3].

# Example 1:


# Input: firstList = [[0,2],[5,10],[13,23],[24,25]], secondList = [[1,5],[8,12],[15,24],[25,26]]
# Output: [[1,2],[5,5],[8,10],[15,23],[24,24],[25,25]]
# Example 2:

# Input: firstList = [[1,3],[5,9]], secondList = []
# Output: []
# Example 3:

# Input: firstList = [], secondList = [[4,8],[10,12]]
# Output: []
# Example 4:

# Input: firstList = [[1,7]], secondList = [[3,10]]
# Output: [[3,7]]

class Solution:
    def intervalIntersection(self, firstList: List[List[int]], secondList: List[List[int]]) -> List[List[int]]:
        # A[1, 3], B[2, 4], intersection when A[0] <= B[1] A[1] >= B[0]
        # [max(A[0], B[0]), min([A[1] ,B[1]])] will be intersection 
        
        a = 0
        b = 0
        output = []
        
        while a < len(firstList) and b < len(secondList):
            # check if intersection exists
            if firstList[a][0] <= secondList[b][1] and firstList[a][1] >= secondList[b][0]:
                # add intersection to output
                output.append([max(firstList[a][0], secondList[b][0]), min(firstList[a][1], secondList[b][1])])
            
            # increment the pointer of subarray that has a lower bound
            if firstList[a][1] <= secondList[b][1]:
                a += 1
            else:
                b += 1
        return output

INTERVAL
# 435. Non-overlapping Intervals
# Medium

# 1925

# 52

# Add to List

# Share
# Given a collection of intervals, find the minimum number of intervals you need to remove to make the rest of the intervals non-overlapping.


# Example 1:

# Input: [[1,2],[2,3],[3,4],[1,3]]
# Output: 1
# Explanation: [1,3] can be removed and the rest of intervals are non-overlapping.
# Example 2:

# Input: [[1,2],[1,2],[1,2]]
# Output: 2
# Explanation: You need to remove two [1,2] to make the rest of intervals non-overlapping.
# Example 3:

# Input: [[1,2],[2,3]]
# Output: 0
# Explanation: You don't need to remove any of the intervals since they're already non-overlapping.

class Solution:
    def eraseOverlapIntervals(self, intervals: List[List[int]]) -> int:
        left = 0
        right = 1
        count = 0 
        intervals.sort(key=lambda x:x[0])
        while right < len(intervals):
            # non-interval case
            if intervals[left][1] <= intervals[right][0]:
                left = right
                right += 1
            # right interval remove case
            elif intervals[left][1] <= intervals[right][1]:
                count += 1
                right += 1
            # left interval remove case
            elif intervals[left][1] >= intervals[right][1]:
                count += 1
                left = right
                right += 1
        return count
            



# 300. Longest Increasing Subsequence
# Given an unsorted array of integers, find the length of longest increasing subsequence.

# Example:

# Input: [10,9,2,5,3,7,101,18]
# Output: 4 
# Explanation: The longest increasing subsequence is [2,3,7,101], therefore the length is 4. 

def lengthOfLIS(self, nums: List[int]) -> int:
    dp = [1]*len(nums)
    output = -1
    
    if len(nums) <= 1:
        return len(nums)
    
    for i in range(1, len(nums)):
        for j in range(i):
            if nums[j] < nums[i]: # the logic to find longest subsequence, prev value < curr
                dp[i] = max(dp[i], dp[j] + 1) # dp[j] + 1 is updating the subsequence
        output = max(output, dp[i])
    return output

# 443. String Compression

# Given an array of characters, compress it in-place.

# The length after compression must always be smaller than or equal to the original array.

# Every element of the array should be a character (not int) of length 1.

# After you are done modifying the input array in-place, return the new length of the array.

# Example 1:

# Input:
# ["a","a","b","b","c","c","c"]

# Output:
# Return 6, and the first 6 characters of the input array should be: ["a","2","b","2","c","3"]

# Explanation:
# "aa" is replaced by "a2". "bb" is replaced by "b2". "ccc" is replaced by "c3".
 

# Example 2:

# Input:
# ["a"]

# Output:
# Return 1, and the first 1 characters of the input array should be: ["a"]

# Explanation:
# Nothing is replaced.

def compress(self, chars: List[str]) -> int:
    # have 2 pointers, i and j traverse the array or char
    # as long as char[i] == char[j], increment j, j - i = char count
    # case: when count is more than 1 digit, so 13 -> 1, 3
    # case: only 1 element in array, [a] -> [a, 1] WRONG
    index = 0
    i = 0
    
    while i < len(chars):
        j = i
        while j < len(chars) and chars[i] == chars[j]:
            j += 1
        
        chars[index] = chars[i]
        index+=1
        if j - i > 1: # so that output array length is less than or equal to original array-> ['a'] does not turn into ['a', '1']
            char_count = str(j - i)
            for char in char_count: # make sure that 13 -> 1, 3
                chars[index] = char
                index+=1
        i = j
    return index

# Q: Reverse a singly linked list.

# Example:

# Input: 1->2->3->4->5->NULL
# Output: 5->4->3->2->1->NULL

# iterative solution
class Solution:
    def reverseList(self, head: ListNode) -> ListNode:
        prev = None
        while head:
            temp = head # temporary variable to keep track of head node
            head = head.next # iterate to the next head node
            temp.next = prev # move the current tmp pointer to prev
            prev = temp # prev then gets updated to tempß
        return prev

# recursive solution
class Solution:
    def reverseList(self, head: ListNode) -> ListNode:
        if head == None, or head.next == None:
            return head
        s = self.reverseList(head.next)
        head.next.next = head
        head.next = None
        return s


# 876. Middle of the Linked List
# Easy

# 2159

# 72

# Add to List

# Share
# Given a non-empty, singly linked list with head node head, return a middle node of linked list.

# If there are two middle nodes, return the second middle node.

# 2 pointer, slow and fast pointer method
class Solution:
    def middleNode(self, head: ListNode) -> ListNode:
        # 1->2->3->4, return 3
        # 1->2->3->4->5, return 3
        slow = head
        fast = head
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
        return slow

# 234. Palindrome Linked List
# Easy

# 4794

# 429

# Add to List

# Share
# Given the head of a singly linked list, return true if it is a palindrome.

 

# Example 1:


# Input: head = [1,2,2,1]
# Output: true
# Example 2:


# Input: head = [1,2]
# Output: false

# combination of finding middle of linked list, reversing linked list, then palindrome validation using 2 pointers
class Solution:
    def isPalindrome(self, head: ListNode) -> bool:
        # find the middle of the likedlist
        # from the middle, reverse the right half of the linked list
        # evaluate each node from beginning and end to see if its a palindrome
        
        slow = fast = head
        # finding the middle of linekdlist
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
        prev = None
        # 1->2->3->2->1, slow = 3
        # reversing the linked list for the "right half"
        while slow:
            tmp = slow
            slow = slow.next
            tmp.next = prev
            prev = tmp
        # now head wil be first node, prev will be last node
        while head and prev:
            if head.val != prev.val:
                return False
            head = head.next
            prev = prev.next
        return True


# 2. Add Two Numbers
# Medium

# 11215

# 2681

# Add to List

# Share
# You are given two non-empty linked lists representing two non-negative integers. The digits are stored in reverse order, and each of their nodes contains a single digit. Add the two numbers and return the sum as a linked list.

# You may assume the two numbers do not contain any leading zero, except the number 0 itself.

 

# Example 1:


# Input: l1 = [2,4,3], l2 = [5,6,4]
# Output: [7,0,8]
# Explanation: 342 + 465 = 807.
# Example 2:

# Input: l1 = [0], l2 = [0]
# Output: [0]
# Example 3:

# Input: l1 = [9,9,9,9,9,9,9], l2 = [9,9,9,9]
# Output: [8,9,9,9,0,0,0,1]

# reversing both linked lists, building an integer sum off the values, taking each decimal place and building a new linked list, then returning the reverse of it
class Solution:
    def addTwoNumbers(self, l1: ListNode, l2: ListNode) -> ListNode:
        first = 0
        second = 0
        def reverse(head):
            prev = None
            while head:
                tmp = head
                head = head.next
                tmp.next = prev
                prev = tmp
            return prev
        l1 = reverse(l1)
        l2 = reverse(l2)
        while l1:
            first = first*10 + l1.val 
            l1 = l1.next
        while l2:
            second = second*10 + l2.val 
            l2 = l2.next
        third = first + second
        third = str(third)
        l3 = ListNode()
        dummy = l3
        for i in range(len(third)):
            l3.next = ListNode(int(third[i]))
            l3 =l3.next
        l3 = dummy.next
        return reverse(l3)

# Q: Given a linked list, determine if it has a cycle in it.

# To represent a cycle in the given linked list, we use an integer pos which represents the position (0-indexed) in the linked list where tail connects to. If pos is -1, then there is no cycle in the linked list.

# Example 1:

# Input: head = [3,2,0,-4], pos = 1
# Output: true
# Explanation: There is a cycle in the linked list, where tail connects to the second node.


# Example 2:

# Input: head = [1], pos = -1
# Output: false
# Explanation: There is no cycle in the linked list.

# O(n) solution both time and space due to using set
def hasCycle(self, head: ListNode) -> bool:
    s = set()
    if head is None or head.next is None:
        return False
    while head:
        s.add(head)
        head = head.next
        if head in s:
            return True
    return False


# O(1) solution at constant space complexity O(1) since only using 2 nodes "fast and slow"
def hasCycle(self, head: ListNode) -> bool:
    # cycle means head.next will never be None
    
    if head is None or head.next is None:
        return False
    
    slow = head
    fast = head.next
    
    while slow != fast:
        if fast is None or fast.next is None:
            return False
        else:
            slow = slow.next
            fast = fast.next.next
    
    return True

# 203. Remove Linked List Elements
# Easy

# 2513

# 122

# Add to List

# Share
# Given the head of a linked list and an integer val, remove all the nodes of the linked list that has Node.val == val, and return the new head.

 

# Example 1:


# Input: head = [1,2,6,3,4,5,6], val = 6
# Output: [1,2,3,4,5]
# Example 2:

# Input: head = [], val = 1
# Output: []
# Example 3:

# Input: head = [7,7,7,7], val = 7
# Output: []
class Solution:
    def removeElements(self, head: ListNode, val: int) -> ListNode:
        # handles when the first head node is the target val to delete
        # head = [7,7,7,7], target val = 7
        while head and head.val == val:
            head = head.next
        
        curr_node = head
        while curr_node and curr_node.next:
            if curr_node.next.val == val:
                # changing curr_node pointer by 2 nodes
                curr_node.next = curr_node.next.next
            # handles edge case where [1, 2, 2, 1] and target val is 2
            # if we just change curr_node pointer by 2, curr_node becomes 2
            # which is a node we need to remove
            while curr_node.next and curr_node.next.val == val:
                curr_node.next = curr_node.next.next
            curr_node = curr_node.next
        return head

# 83. Remove Duplicates from Sorted List
# Easy

# 2378

# 143

# Add to List

# Share
# Given the head of a sorted linked list, delete all duplicates such that each element appears only once. Return the linked list sorted as well.

 

# Example 1:


# Input: head = [1,1,2]
# Output: [1,2]
# Example 2:


# Input: head = [1,1,2,3,3]
# Output: [1,2,3]

class Solution:
    def deleteDuplicates(self, head: ListNode) -> ListNode:
        if not head:
            return head
        origin = head
        while head and head.next:
            if head.val == head.next.val: # 1->1->2->3->3 turn into 1->2->3
                head.next = head.next.next
            else:
                head = head.next
        return origin

# Q: Merge two sorted linked lists and return it as a new list. The new list should be made by splicing together the nodes of the first two lists.

# Example:

# Input: 1->2->4, 1->3->4
# Output: 1->1->2->3->4->4

 def mergeTwoLists(self, l1: ListNode, l2: ListNode) -> ListNode:
        
    dummy = ListNode(0)
    head = dummy
    
    while l1 != None and l2 != None:
        if l1.val > l2.val:
            dummy.next = l2
            l2 = l2.next
        else: # takes care of when l1.val < l2.val AND l1.val == l2.val
            dummy.next = l1
            l1 = l1.next
            
        dummy = dummy.next  # most important part
        
    if l1 != None: # append the rest of the linked list 
        dummy.next = l1
    else:
        dummy.next = l2
    return head.next

# same as above
class Solution:
    def mergeTwoLists(self, l1: ListNode, l2: ListNode) -> ListNode:
        # traverse both lists and compare node vals,  
        # if l1.val > l2.val, l2.next = l1, l2 = l2.next and vice versa
        
        root = dummy = ListNode(0)
        # root or dummy is 0, thus we need the next node that doesn't exist yet
        if not l1 and not l2:
            return root.next
            
        while l1 and l2:
            if l1.val >= l2.val:
                dummy.next = ListNode(l2.val)
                l2 = l2.next
            elif l1.val <= l2.val:
                dummy.next = ListNode(l1.val)
                l1 = l1.next
            dummy = dummy.next
        
        if not l1:
            dummy.next = l2
        else:
            dummy.next = l1
        return root.next

# Q: Merge k sorted linked lists and return it as one sorted list. Analyze and describe its complexity.

# Example:

# Input:
# [
#   1->4->5,
#   1->3->4,
#   2->6
# ]
# Output: 1->1->2->3->4->4->5->6

# brute force 
# time: O(nlogn) since collecting nodes is O(n), sorting is O(nlogn)
# space: O(n) since creating new linked list  costs O(n) space
def mergeKLists(self, lists: List[ListNode]) -> ListNode:
    nodes = []
    head = point = ListNode(0)
    
    for l in lists:
        while l:
            nodes.append(l.val)
            l = l.next
    
    for node in sorted(nodes):
        point.next = ListNode(node) # important to turn datatype back into ListNode from int
        point = point.next
    
    return head.next

BFS
# 102. Binary Tree Level Order Traversal
# Medium

# 4345

# 103

# Add to List

# Share
# Given the root of a binary tree, return the level order traversal of its nodes' values. (i.e., from left to right, level by level).

 

# Example 1:


# Input: root = [3,9,20,null,null,15,7]
# Output: [[3],[9,20],[15,7]]
# Example 2:

# Input: root = [1]
# Output: [[1]]
# Example 3:

# Input: root = []
# Output: []

# BFS pattern
class Solution:
    def levelOrder(self, root: TreeNode) -> List[List[int]]:
        # check if tree even has a root node
        if not root:
            return []
        # deque allows append/pop from start/end with O(1) time complexity
        queue = deque([root])
        output = []
        
        # as long as queue isn't empty
        while queue:
            curr_level = []
            size = len(queue)
            for i in range(size):
                # popping from left since order is left then right 
                node = queue.popleft()
                # checking to see if node has children on left side
                if node.left:
                    queue.append(node.left)
                # checking to see if node has children on right side
                if node.right:
                    queue.append(node.right)
                # append the nodes on current level of tree
                curr_level.append(node.val)
            # append all nodes on current level to the output
            output.append(curr_level)
        return output

BFS
# 103. z
# Medium

# 3189

# 125

# Add to List

# Share
# Given the root of a binary tree, return the zigzag level order traversal of its nodes' values. (i.e., from left to right, then right to left for the next level and alternate between).

 

# Example 1:


# Input: root = [3,9,20,null,null,15,7]
# Output: [[3],[20,9],[15,7]]
# Example 2:

# Input: root = [1]
# Output: [[1]]
# Example 3:

# Input: root = []
# Output: []

# BFS like above question but alternating how we append each level of nodes 
class Solution:
    def zigzagLevelOrder(self, root: TreeNode) -> List[List[int]]:
        if not root:
            return []
        queue = deque([root])
        output = []
        level = 0
        
        while queue:   
            size = len(queue)
            curr_level = deque([]) # so that we can use appendleft during the "odd" levels where we add nodes right -> left
            for i in range(size):
                curr_node = queue.popleft()
                if curr_node.left:
                    queue.append(curr_node.left)
                if curr_node.right:
                    queue.append(curr_node.right)
                if level % 2 == 0:
                    curr_level.append(curr_node.val)
                elif level % 2 != 0:
                    curr_level.appendleft(curr_node.val)
            level+= 1
            output.append(curr_level)
        return output

BFS
# 107. Binary Tree Level Order Traversal II
# Medium

# 2054

# 239

# Add to List

# Share
# Given the root of a binary tree, return the bottom-up level order traversal of its nodes' values. (i.e., from left to right, level by level from leaf to root).

 

# Example 1:


# Input: root = [3,9,20,null,null,15,7]
# Output: [[15,7],[9,20],[3]]
# Example 2:

# Input: root = [1]
# Output: [[1]]
# Example 3:

# Input: root = []
# Output: []

# similar solution to levelordertraversal above but using appendleft when populating the output array 
class Solution:
    def levelOrderBottom(self, root: TreeNode) -> List[List[int]]:
        if not root:
            return []
        queue = deque([root])
        output = deque([])
        
        while queue:
            size = len(queue)
            curr_level = []
            for i in range(size):
                curr_node = queue.popleft()
                if curr_node.left:
                    queue.append(curr_node.left)
                if curr_node.right:
                    queue.append(curr_node.right)
                curr_level.append(curr_node.val)
            output.appendleft(curr_level)
        return output
BFS
# 637. Average of Levels in Binary Tree
# Easy

# 1921

# 200

# Add to List

# Share
# Given the root of a binary tree, return the average value of the nodes on each level in the form of an array. Answers within 10-5 of the actual answer will be accepted.
 

# Example 1:


# Input: root = [3,9,20,null,15,7]
# Output: [3.00000,14.50000,11.00000]
# Explanation: The average value of nodes on level 0 is 3, on level 1 is 14.5, and on level 2 is 11.
# Hence return [3, 14.5, 11].
# Example 2:


# Input: root = [3,9,20,15,7]
# Output: [3.00000,14.50000,11.00000]

class Solution:
    def averageOfLevels(self, root: TreeNode) -> List[float]:
        output = []
        queue = deque([root])
        while queue:
            curr_level = []
            avg_sum = 0
            size = len(queue)
            for i in range(size):
                curr_node = queue.popleft()
                if curr_node.left:
                    queue.append(curr_node.left)
                if curr_node.right:
                    queue.append(curr_node.right)
                curr_level.append(curr_node.val)
            print(curr_level)
            for i in range(len(curr_level)):
                avg_sum += curr_level[i]
            avg_sum = avg_sum / len(curr_level)
            output.append(avg_sum)
        return output

# 111. Minimum Depth of Binary Tree
# Easy

# 2240

# 802

# Add to List

# Share
# Given a binary tree, find its minimum depth.

# The minimum depth is the number of nodes along the shortest path from the root node down to the nearest leaf node.

# Note: A leaf is a node with no children.

 

# Example 1:


# Input: root = [3,9,20,null,null,15,7]
# Output: 2
# Example 2:

# Input: root = [2,null,3,null,4,null,5,null,6]
# Output: 5
BFS
class Solution:
    def minDepth(self, root: TreeNode) -> int:
        # traverse tree and check if a node is a leaf
        # leaf node has no right or left child
        
        if not root:
            return 0
        queue = deque([root])
        min_depth = float('inf')
        curr_depth = 0
        while queue:
            size = len(queue)
            curr_depth += 1
            for i in range(size):
                curr_node = queue.popleft()
                if curr_node.left:
                    queue.append(curr_node.left)
                if curr_node.right:
                    queue.append(curr_node.right)
                if not curr_node.left and not curr_node.right:
                    min_depth = min(min_depth, curr_depth)
        return min_depth
BFS modified solution from above
class Solution:
    def minDepth(self, root: TreeNode) -> int:
        # traverse tree and check if a node is a leaf
        # leaf node has no right or left child
        
        if not root:
            return 0
        queue = deque([(root, 1)])
        while queue:
            size = len(queue)
            for i in range(size):
                curr_node, level = queue.popleft()      
                if curr_node.left:
                    queue.append((curr_node.left, level + 1))
                if curr_node.right:
                    queue.append((curr_node.right, level + 1))
                if not curr_node.left and not curr_node.right:
                    return level

# 94. Binary Tree Inorder Traversal
# Medium

# 4424

# 197

# Add to List

# Share
# Given the root of a binary tree, return the inorder traversal of its nodes' values.

 

# Example 1:


# Input: root = [1,null,2,3]
# Output: [1,3,2]
# Example 2:

# Input: root = []
# Output: []
# Example 3:

# Input: root = [1]
# Output: [1]

DFS

class Solution:
    def inorderTraversal(self, root: TreeNode) -> List[int]:
        if not root:
            return []
        stack = [(root, False)]
        output = []
        
        while stack:
            curr_node, visited = stack.pop()
            if curr_node: 
                if visited:
                    output.append(curr_node.val)
                else:
                    stack.append((curr_node.right, False))
                    stack.append((curr_node, True))
                    stack.append((curr_node.left, False))
        return output

# iterative solution, in order traversal is left, root, right
class Solution:
    def inorderTraversal(self, root: TreeNode) -> List[int]:
        if not root:
            return []
        
        stack = []
        output = []
        while stack or root:
            if root:
                stack.append(root)
                root = root.left
            else:
                curr_node = stack.pop()
                output.append(curr_node.val)
                root = curr_node.right
        return output



# 144. Binary Tree Preorder Traversal
# Medium

# 2149

# 87

# Add to List

# Share
# Given the root of a binary tree, return the preorder traversal of its nodes' values.

 

# Example 1:


# Input: root = [1,null,2,3]
# Output: [1,2,3]
# Example 2:

# Input: root = []
# Output: []
# Example 3:

# Input: root = [1]
# Output: [1]

class Solution:
    def preorderTraversal(self, root: TreeNode) -> List[int]:
        if not root:
            return []
        stack = [(root, False)]
        output = []
        
        while stack:
            curr_node, visited = stack.pop()
            if curr_node: 
                if visited:
                    output.append(curr_node.val)
                else:
                    stack.append((curr_node.right, False))
                    stack.append((curr_node.left, False))
                    stack.append((curr_node, True))
        return output

DFS 
class Solution:
    def preorderTraversal(self, root: TreeNode) -> List[int]:
        # root, left, right
        if not root:
            return []
        output = []
        stack = []
        
        while stack or root:
            if root:
                stack.append(root) 
                output.append(root.val)
                root = root.left
            else:
                root = stack.pop() 
                root = root.right
        return output

# 145. Binary Tree Postorder Traversal
# Medium

# 2462

# 113

# Add to List

# Share
# Given the root of a binary tree, return the postorder traversal of its nodes' values.

 

# Example 1:


# Input: root = [1,null,2,3]
# Output: [3,2,1]
# Example 2:

# Input: root = []
# Output: []
# Example 3:

# Input: root = [1]
# Output: [1]

DFS

class Solution:
    def postorderTraversal(self, root: TreeNode) -> List[int]:
        traversal, stack = [], [(root, False)]
        # [1, 2, None] -> [1, 2] -> [1, 2, None, 3] -> [1, 2, None, 3, None, None]
        # [1, 2, None, 3, None] -> [1, 2, None, 3] -> [1, 2, None] -> [1, 2] -> [1] -> []
        while stack: 
            node, visited = stack.pop()
            if node:
                if visited:
                    # add to result if visited
                    traversal.append(node.val)
                else:
                    # post-order
                    stack.append((node, True))
                    stack.append((node.right, False))
                    stack.append((node.left, False))
        return traversal


# 104. Maximum Depth of Binary Tree
# Given a binary tree, find its maximum depth.

# The maximum depth is the number of nodes along the longest path from the root node down to the farthest leaf node.

# Note: A leaf is a node with no children.

# Example:

# Given binary tree [3,9,20,null,null,15,7],

#     3
#    / \
#   9  20
#     /  \
#    15   7
# return its depth = 3.


Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def maxDepth(self, root: TreeNode) -> int:
        if root is None:
            return 0
        else:
            return 1 + max(self.maxDepth(root.left), self.maxDepth(root.right))

# iterative bfs, building an array of nodes per level and returning the length of that
class Solution:
    def maxDepth(self, root: TreeNode) -> int:
        if not root:
            return 0
        queue = deque([root])
        nodes = []
        while queue:
            size = len(queue)
            curr_level = []
            for i in range(size):
                curr_node = queue.popleft()
                if curr_node.left:
                    queue.append(curr_node.left)
                if curr_node.right:
                    queue.append(curr_node.right)
                curr_level.append(curr_node.val)
            nodes.append(curr_level)
        return len(nodes)

# iterative bfs using a counter variable
class Solution:
    def maxDepth(self, root: TreeNode) -> int:
        if not root:
            return 0
        queue = deque([root])
        depth = 0
        while queue:
            size = len(queue)
            for i in range(size):
                curr_node = queue.popleft()
                if curr_node.left:
                    queue.append(curr_node.left)
                if curr_node.right:
                    queue.append(curr_node.right)
            depth += 1
        return depth
# 100. Same Tree
# Given two binary trees, write a function to check if they are the same or not.

# Two binary trees are considered the same if they are structurally identical and the nodes have the same value.

# Example 1:

# Input:     1         1
#           / \       / \
#          2   3     2   3

#         [1,2,3],   [1,2,3]

# Output: true
# Example 2:

# Input:     1         1
#           /           \
#          2             2

#         [1,2],     [1,null,2]

# Output: false

class Solution:
    def isSameTree(self, p: TreeNode, q: TreeNode) -> bool:
        if p is None and q is None:
            return True
        if p is None or q is None:
            return False
        if p.val != q.val:
            return False
        return self.isSameTree(p.left, q.left) and self.isSameTree(p.right, q.right)

BFS iterative solution
class Solution:
    def isSameTree(self, p: TreeNode, q: TreeNode) -> bool:
        def check(p, q):
            # both are None
            if not p and not q:
                return True
            # one is None
            if not p or not q:
                return False
            if p.val != q.val:
                return False
            return True
        queue = deque([(p, q)])
        while queue:
            p, q = queue.popleft()
            if not check(p, q):
                return False
            if p:
                queue.append((p.left, q.left))
                queue.append((p.right, q.right))
        return True

class Solution:
    def isSameTree(self, p: TreeNode, q: TreeNode) -> bool:
        queue = deque([(p, q)])
        while queue:
            p, q = queue.popleft()
            if not p and not q:
                continue
            elif not p or not q:
                return False
            else:
                if p.val != q.val:
                    return False
                if p:
                    queue.append((p.left, q.left))
                    queue.append((p.right, q.right))
        return True
        
# 226. Invert Binary Tree
# Invert a binary tree.

# Example:

# Input:

#      4
#    /   \
#   2     7
#  / \   / \
# 1   3 6   9
# Output:

#      4
#    /   \
#   7     2
#  / \   / \
# 9   6 3   1
class Solution:
    def invertTree(self, root: TreeNode) -> TreeNode:
        if root is None:
            return None
        
        left = self.invertTree(root.left)
        right = self.invertTree(root.right)
        
        root.left = right
        root.right = left
        
        return root
# iterative solution using BFS
class Solution:
    def invertTree(self, root: TreeNode) -> TreeNode:
        if root is None:
            return None
        queue = deque([root])
        nodes = deque([])
        while queue:
            size = len(queue)
            for i in range(size):
                curr_node = queue.popleft()
                if curr_node:
                    tmp_node = curr_node.left
                    curr_node.left = curr_node.right
                    curr_node.right = tmp_node
                    if curr_node.left:
                        queue.append(curr_node.left)
                    if curr_node.right:
                        queue.append(curr_node.right)
        return root

# 572. Subtree of Another Tree
# Given two non-empty binary trees s and t, check whether tree t has exactly the same structure and node values with a subtree of s. 
# A subtree of s is a tree consists of a node in s and all of this node's descendants. The tree s could also be considered as a subtree of itself.

# Example 1:
# Given tree s:

#      3
#     / \
#    4   5
#   / \
#  1   2
# Given tree t:
#    4 
#   / \
#  1   2
# Return true, because t has the same structure and node values with a subtree of s.
 

# Example 2:
# Given tree s:

#      3
#     / \
#    4   5
#   / \
#  1   2
#     /
#    0
# Given tree t:
#    4
#   / \
#  1   2
# Return false.
class Solution:
    def isSubtree(self, s: TreeNode, t: TreeNode) -> bool:
        if s is None:
            return False
        elif self.isSame(s, t):
            return True
        else:
            return self.isSubtree(s.left, t) or self.isSubtree(s.right, t)
        
    def isSame(self, a, b):
        if a is None and b is None:
            return True
        if a is None or b is None:
            return False
        if a.val != b.val:
            return False
        return self.isSame(a.left, b.left) and self.isSame(a.right, b.right)

# iteartive solution using queue
class Solution:
    def isSubtree(self, s: TreeNode, t: TreeNode) -> bool:
        def sameTree(s, t):
            queue = deque([(s, t)])
            while queue:
                s, t = queue.popleft()
                if not s and not t:
                    continue
                if not s or not t:
                    return False
                if s.val != t.val:
                    return False
                if s:
                    queue.append((s.left, t.left))
                    queue.append((s.right, t.right))
            return True
        queue = deque([s])
        while queue:
            s = queue.popleft()
            if sameTree(s, t):
                return True
            if s:
                queue.append(s.left)
                queue.append(s.right)
        return False


# 112. Path Sum
# Easy

# 2955

# 594

# Add to List

# Share
# Given the root of a binary tree and an integer targetSum, return true if the tree has a root-to-leaf path such that adding up all the values along the path equals targetSum.

# A leaf is a node with no children.

 

# Example 1:


# Input: root = [5,4,8,11,null,13,4,7,2,null,null,null,1], targetSum = 22
# Output: true
# Example 2:


# Input: root = [1,2,3], targetSum = 5
# Output: false
# Example 3:

# Input: root = [1,2], targetSum = 0
# Output: false

# BFS, traverse tree and keep track of node value sums as you traverse
class Solution:
    def hasPathSum(self, root: TreeNode, targetSum: int) -> bool:
        if not root:
            return False
        queue = deque([(root, root.val)])
        while queue:
            curr_node, val = queue.popleft()
            # condition to check if lead node and if the paths sum is target sum
            if not curr_node.left and not curr_node.right and val == targetSum:
                return True
            if curr_node.left:
                queue.append((curr_node.left, val + curr_node.left.val))
            if curr_node.right:
                queue.append((curr_node.right, val + curr_node.right.val))
        return False


# 617. Merge Two Binary Trees
# Easy

# 4113

# 193

# Add to List

# Share
# You are given two binary trees root1 and root2.

# Imagine that when you put one of them to cover the other, some nodes of the two trees are overlapped while the others are not. You need to merge the two trees into a new binary tree. The merge rule is that if two nodes overlap, then sum node values up as the new value of the merged node. Otherwise, the NOT null node will be used as the node of the new tree.

# Return the merged tree.

# Note: The merging process must start from the root nodes of both trees.

 

# Example 1:


# Input: root1 = [1,3,2,5], root2 = [2,1,3,null,4,null,7]
# Output: [3,4,5,5,4,null,7]
# Example 2:

# Input: root1 = [1], root2 = [1,2]
# Output: [2,2]

class Solution(object):
    def mergeTrees(self, t1, t2):
        """
        :type t1: TreeNode
        :type t2: TreeNode
        :rtype: TreeNode
        """
        if not t1:
            return t2
        queue = deque([(t1,t2)])
        dummy = t1
        while queue:
            t1, t2 = queue.popleft()
            if not t1 or not t2:
                continue
            else:
                t1.val += t2.val
            if not t1.left:
                t1.left = t2.left
            else:
                queue.append((t1.left, t2.left))                
            if not t1.right:
                t1.right = t2.right
            else:
                queue.append((t1.right, t2.right))
        return dummy

# 235. Lowest Common Ancestor of a Binary Search Tree
# Easy

# 2960

# 130

# Add to List

# Share
# Given a binary search tree (BST), find the lowest common ancestor (LCA) of two given nodes in the BST.

# According to the definition of LCA on Wikipedia: “The lowest common ancestor is defined between two nodes p and q as the lowest node in T that has both p and q as descendants (where we allow a node to be a descendant of itself).”

 

# Example 1:


# Input: root = [6,2,8,0,4,7,9,null,null,3,5], p = 2, q = 8
# Output: 6
# Explanation: The LCA of nodes 2 and 8 is 6.
# Example 2:


# Input: root = [6,2,8,0,4,7,9,null,null,3,5], p = 2, q = 4
# Output: 2
# Explanation: The LCA of nodes 2 and 4 is 2, since a node can be a descendant of itself according to the LCA definition.
# Example 3:

# Input: root = [2,1], p = 2, q = 1
# Output: 2

class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        # left node < parent, right node > parent for BST
        curr_node = root
        while curr_node:
            parent = curr_node.val
            if p.val > parent and q.val > parent:
                curr_node = curr_node.right
            elif p.val < parent and q.val < parent:
                curr_node = curr_node.left
            else:
                return curr_node
            

# 98. Validate Binary Search Tree
# Given a binary tree, determine if it is a valid binary search tree (BST).

# Assume a BST is defined as follows:

# The left subtree of a node contains only nodes with keys less than the node's key.
# The right subtree of a node contains only nodes with keys greater than the node's key.
# Both the left and right subtrees must also be binary search trees.
 

# Example 1:

#     2
#    / \
#   1   3

# Input: [2,1,3]
# Output: true
# Example 2:

#     5
#    / \
#   1   4
#      / \
#     3   6

# Input: [5,1,4,null,null,3,6]
# Output: false
# Explanation: The root node's value is 5 but its right child's value is 4.
class Solution:
    def isValidBST(self, root: TreeNode) -> bool:
        return self.validate(root, None, None)
    
    def validate(self, root, max, min):
        if root is None:
            return True
        elif max != None and root.val >= max or min != None and root.val <= min:
            return False
        else:
            return self.validate(root.left, root.val, min) and self.validate(root.right, max, root.val)
        


# Q: Given a string containing just the characters '(', ')', '{', '}', '[' and ']', determine if the input string is valid.

# An input string is valid if:

# Open brackets must be closed by the same type of brackets.
# Open brackets must be closed in the correct order.
# Note that an empty string is also considered valid.

# Example 1:

# Input: "()"
# Output: true
# Example 2:

# Input: "()[]{}"
# Output: true
# Example 3:

# Input: "(]"
# Output: false

def isValid(s):
    stack = []
    mapping = {")": "(", "}": "{", "]": "["}

    for char in s:
        if char in mapping:
            top_ele = stack.pop() if stack else "#"
            if mapping[char] != top_ele:
                return False
        else:
            stack.append(char)
    return not stack


HEAP

# 295. Find Median from Data Stream
# Hard

# 3944

# 72

# Add to List

# Share
# The median is the middle value in an ordered integer list. If the size of the list is even, there is no middle value and the median is the mean of the two middle values.

# For example, for arr = [2,3,4], the median is 3.
# For example, for arr = [2,3], the median is (2 + 3) / 2 = 2.5.
# Implement the MedianFinder class:

# MedianFinder() initializes the MedianFinder object.
# void addNum(int num) adds the integer num from the data stream to the data structure.
# double findMedian() returns the median of all elements so far. Answers within 10-5 of the actual answer will be accepted.
 

# Example 1:

# Input
# ["MedianFinder", "addNum", "addNum", "findMedian", "addNum", "findMedian"]
# [[], [1], [2], [], [3], []]
# Output
# [null, null, null, 1.5, null, 2.0]

# Explanation
# MedianFinder medianFinder = new MedianFinder();
# medianFinder.addNum(1);    // arr = [1]
# medianFinder.addNum(2);    // arr = [1, 2]
# medianFinder.findMedian(); // return 1.5 (i.e., (1 + 2) / 2)
# medianFinder.addNum(3);    // arr[1, 2, 3]
# medianFinder.findMedian(); // return 2.0

class MedianFinder:

    def __init__(self):
        """
        initialize your data structure here.
        """
        # smaller half of list, the max_heap
        self.small = []
        # bigger half of list, the min_heap
        self.big = []

    def addNum(self, num: int) -> None:
        if len(self.small) == len(self.big):
            # adding to maxheap, thus -num (since python only has minheap, we make it negative number), then adding to minheap, so -heappushpop to make it positive 
            heappush(self.big, -heappushpop(self.small, -num))
        else:
            heappush(self.small, -heappushpop(self.big, num))

    def findMedian(self) -> float:
        if len(self.small) == len(self.big):
            return float(self.big[0] - self.small[0]) / 2
        else:
            return float(self.big[0])

# 373. Find K Pairs with Smallest Sums
# Medium

# 1860

# 126

# Add to List

# Share
# You are given two integer arrays nums1 and nums2 sorted in ascending order and an integer k.

# Define a pair (u, v) which consists of one element from the first array and one element from the second array.

# Return the k pairs (u1, v1), (u2, v2), ..., (uk, vk) with the smallest sums.

 

# Example 1:

# Input: nums1 = [1,7,11], nums2 = [2,4,6], k = 3
# Output: [[1,2],[1,4],[1,6]]
# Explanation: The first 3 pairs are returned from the sequence: [1,2],[1,4],[1,6],[7,2],[7,4],[11,2],[7,6],[11,4],[11,6]
# Example 2:

# Input: nums1 = [1,1,2], nums2 = [1,2,3], k = 2
# Output: [[1,1],[1,1]]
# Explanation: The first 2 pairs are returned from the sequence: [1,1],[1,1],[1,2],[2,1],[1,2],[2,2],[1,3],[1,3],[2,3]
# Example 3:

# Input: nums1 = [1,2], nums2 = [3], k = 3
# Output: [[1,3],[2,3]]
# Explanation: All possible pairs are returned from the sequence: [1,3],[2,3]

# Brute force, O(n^2)
class Solution:
    def kSmallestPairs(self, nums1: List[int], nums2: List[int], k: int) -> List[List[int]]:
        sums = []
        output = []
        for num_1 in nums1:
            for num_2 in nums2:
                sums.append([num_1, num_2])
        sorted(sums, key=lambda x:sum(x))
        i = 0
        while i < k and i < len(sums):
            output.append(sums[i])
            i+=1
        return output

# O(nlogn) solution using a heap 
class Solution:
    def kSmallestPairs(self, nums1: List[int], nums2: List[int], k: int) -> List[List[int]]:
        if not nums1 and not nums2:
            return []
        # to push (sum, index1, index2) in a minheap
        heap = []
        # output array
        output = []
        # keep track of index that's been visited
        visited = set()
        # push intial sum and their index into heap
        heappush(heap, (nums1[0] + nums2[0], 0, 0))
        # mark the current indexes as visited
        visited.add((0, 0))
        # exit condition since len(output) cant exceed k and elements in heap
        while len(output) < k and heap:
            # the sum and pair of current index as a tuple
            val = heappop(heap)
            output.append((nums1[val[1]], nums2[val[2]]))
            # check if the next index of nums1 is within bounds and if pair of index not visited
            if val[1] + 1 < len(nums1) and (val[1] + 1, val[2]) not in visited:
                # push to heap and mark as visited
                heappush(heap, (nums1[val[1] + 1] + nums2[val[2]], val[1] + 1, val[2]))
                visited.add((val[1] + 1, val[2]))
            #  check if the next index of nums2 is within bounds and if pair of index not visited
            if val[2] + 1 < len(nums2) and (val[1], val[2] + 1) not in visited:
                # push to heap and mark as visited
                heappush(heap, (nums1[val[1]] + nums2[val[2] + 1], val[1], val[2] + 1))
                visited.add((val[1], val[2] + 1))
            # print(visited)
        return output

# 347. Top K Frequent Elements
# Medium

# 4702

# 265

# Add to List

# Share
# Given an integer array nums and an integer k, return the k most frequent elements. You may return the answer in any order.

 

# Example 1:

# Input: nums = [1,1,1,2,2,3], k = 2
# Output: [1,2]
# Example 2:

# Input: nums = [1], k = 1
# Output: [1]

# brute force
class Solution:
    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        nums_map = {}
        for i in range(len(nums)):
            if nums[i] not in nums_map:
                nums_map[nums[i]] = 1
            else:
                nums_map[nums[i]] += 1
        print(nums_map)
        output = set()
        max_val = 0
        max_freq = 0
        while k > 0:
            for key, val in nums_map.items():
                if val > max_freq and key not in output:
                    max_freq = val
                    max_val = key
            output.add(max_val)
            max_freq = 0
            k -= 1
        output = list(output)
        return output