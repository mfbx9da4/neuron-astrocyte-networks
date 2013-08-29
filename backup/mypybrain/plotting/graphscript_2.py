import matplotlib.pyplot as plt

title2 = 'Test after ' + str(iters) + ' iterations'
plt.title(title2)
plt.ylabel('Node output')
plt.xlabel('Instances')
plt.plot(results, 'xr', linewidth=1.5, label='Results')
plt.plot(targets, 's', color='black', linewidth=3, label='Targets')
plt.legend(loc='lower right')

plt.figure(2)
plt.subplot(121)
plt.title('Top individual error evolution')
plt.ylabel('Inverse error')
plt.xlabel('Iterations')
plt.plot(all_top_mses, '-g', linewidth=1)
plt.subplot(122)
plt.plot(all_avg_mses, '-g', linewidth=1)
plt.title('Population average error evolution')
plt.ylabel('Inverse error')
plt.xlabel('Iterations')

plt.show()